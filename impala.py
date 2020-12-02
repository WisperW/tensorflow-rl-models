# encoding: utf-8
import tensorflow as tf
import numpy as np
lr_a=1e-5
lr_c=3e-5
clip_ratio=0.2
ent_rate = 0.01
clip_rho_threshold = 1
discount_factor = 0.99
class build_agent_gpu():
    def __init__(self, image_shape = (4, 4), a_dim = 4, s1_dim = 20, s2_dim = 10, num_gpus = 4, using_image = False, num_trajectories = 64):
        self._use_image = using_image
        self.a_dim      = a_dim
        self.infer_image      = tf.placeholder(tf.float32, [None, *image_shape, 1])
        self.infer_obs1       = tf.placeholder(tf.float32, [None, s1_dim])
        self.infer_obs2       = tf.placeholder(tf.float32, [None, s2_dim])
        ###
        self.num_trajectories = num_trajectories # same to batch_size
        single_gpu_num_trajectories = num_trajectories//num_gpus # 注意取整
        self.log_old_pi_ph = tf.placeholder(tf.float32,[None, num_trajectories])
        self.act_ph          = tf.placeholder(tf.int32,[None, num_trajectories])
        self.rew_ph          = tf.placeholder(tf.float32,[None, num_trajectories])
        self.done_ph       = tf.placeholder(tf.float32,[None, num_trajectories])
        self.obs1_ph       = tf.placeholder(tf.float32, [None, num_trajectories, s1_dim])
        self.obs2_ph       = tf.placeholder(tf.float32, [None, num_trajectories, s2_dim])
        self.image_ph      = tf.placeholder(tf.float32,[None, num_trajectories, *image_shape, 1])
        # self.last_state_ph = tf.placeholder(tf.float32,[None, num_trajectories, *image_shape, 1])

        ###
        self.ep_reward = tf.placeholder(tf.float32)
        self.critic_bn_flag = tf.placeholder(tf.bool)
        self.eval_bn_flag = tf.placeholder(tf.bool)
        self.targ_bn_flag = tf.placeholder(tf.bool)
        
        kl = []
        grads_a = []
        grads_c = []
        self.a_optimizer = tf.train.AdamOptimizer(lr_a)
        self.c_optimizer = tf.train.AdamOptimizer(lr_c)
        for i in range(num_gpus):
            with tf.device('/gpu:{}'.format(i)):
                single_gpu_obs1         = self.obs1_ph[:, i * single_gpu_num_trajectories:(i + 1) * single_gpu_num_trajectories]
                single_gpu_obs2         = self.obs2_ph[:, i * single_gpu_num_trajectories:(i + 1) * single_gpu_num_trajectories]
                single_gpu_image        = self.image_ph[:, i * single_gpu_num_trajectories:(i + 1) * single_gpu_num_trajectories]
                single_gpu_act_ph       = self.act_ph[:, i * single_gpu_num_trajectories:(i + 1) * single_gpu_num_trajectories]
                single_gpu_rew_ph       = self.rew_ph[:, i * single_gpu_num_trajectories:(i + 1) * single_gpu_num_trajectories]
                single_gpu_done_ph      = self.done_ph[:, i * single_gpu_num_trajectories:(i + 1) * single_gpu_num_trajectories]
                single_gpu_log_old_pi_ph= self.log_old_pi_ph[:, i * single_gpu_num_trajectories:(i + 1) * single_gpu_num_trajectories]
                # single_gpu_ret = self.ret_ph[:, i * single_gpu_num_trajectories:(i + 1) * single_gpu_num_trajectories]

                if not self._use_image:
                    cnn_out_state = tf.constant(1.0)
                else:
                    single_gpu_image_input = tf.reshape(single_gpu_image, [-1, *image_shape, 1]) 
                    with tf.variable_scope('CNN', reuse=tf.AUTO_REUSE):
                        cnn_out_state       = self._build_cnn(single_gpu_image_input ,'pi', self.critic_bn_flag, True)    

                single_gpu_obs1 = tf.reshape(single_gpu_obs1, [-1, s1_dim])
                single_gpu_obs2 = tf.reshape(single_gpu_obs2, [-1, s2_dim])
                with tf.variable_scope('Critic', reuse=tf.AUTO_REUSE):
                    values         = self._build_c(cnn_out_state, single_gpu_obs1, single_gpu_obs2,\
                                                'pi', self.critic_bn_flag, True, single_gpu_num_trajectories, is_infer = False)
                    # shape of values and next_values is [T, single_gpu_num_trajectories]

                with tf.variable_scope('Actor', reuse=tf.AUTO_REUSE):
                    logit      = self._build_a(cnn_out_state, single_gpu_obs1, single_gpu_obs2, 'pi', self.eval_bn_flag, True, single_gpu_num_trajectories, is_infer = False)
                    print('self.logit', logit.shape)
                    log_policy = tf.nn.log_softmax(logit)

                # shape of log_pi = [T, single_gpu_num_trajectories]
                print('self.log_policy', log_policy)
                log_pi     = tf.reduce_sum(tf.one_hot(single_gpu_act_ph, depth=a_dim) *log_policy,axis=2)
                print('log_pi', log_pi.shape)
                log_rhos = log_pi - single_gpu_log_old_pi_ph
                discounts = (1-single_gpu_done_ph) * discount_factor
                # shape of bootstrap_value is (single_gpu_num_trajectories,)
                bootstrap_value = tf.zeros([single_gpu_num_trajectories])
                vs, pg_advantages = self.from_importance_weights(log_rhos, discounts, single_gpu_rew_ph, values, bootstrap_value)
                entropy, z = self.cat_entropy(tf.reshape(log_policy, [-1, self.a_dim]))
                pi_loss = -tf.reduce_mean(log_pi * pg_advantages)-tf.reduce_mean(entropy * ent_rate)
                v_loss  = tf.reduce_mean((vs - values)**2*0.5)
                print('vs', vs.shape)

            if not using_image:
                self.pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/pi')
                self.c_params  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic')
            else:
                self.pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'CNN/pi')+\
                                       tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/pi')
                self.c_params  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'CNN/pi')+\
                                       tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic')
                                    

            grads_a.append(self.a_optimizer.compute_gradients(pi_loss, self.pi_params))
            #(self.get_grads(self.a_optimizer, self.pi_params, pi_loss))
            grads_c.append(self.c_optimizer.compute_gradients(v_loss, self.c_params))
            #(self.get_grads(self.c_optimizer, self.c_params, v_loss))

        grad_a = self.average_gradients(grads_a)
        grad_c = self.average_gradients(grads_c)
        self.a_train = self.a_optimizer.apply_gradients(grad_a)
        self.c_train = self.c_optimizer.apply_gradients(grad_c)
        

        if not self._use_image:
            cnn_out_state = tf.constant(1.0)
        else:
            with tf.variable_scope('CNN', reuse=tf.AUTO_REUSE):
                cnn_out_state       = self._build_cnn(self.infer_image ,'pi', self.critic_bn_flag, True)    
        with tf.variable_scope('Critic', reuse=tf.AUTO_REUSE):
            self.critic = self._build_c(cnn_out_state, self.infer_obs1, self.infer_obs2,\
                                        'pi', self.critic_bn_flag, True, single_gpu_num_trajectories, is_infer = True)
        with tf.variable_scope('Actor', reuse=tf.AUTO_REUSE):
            self.infer_logit         = self._build_a(cnn_out_state, self.infer_obs1, self.infer_obs2, 'pi', self.eval_bn_flag, True, single_gpu_num_trajectories, is_infer = True)
            self.infer_log_prob      = tf.nn.log_softmax(self.infer_logit)
            self.sample_action =tf.squeeze(tf.multinomial(self.infer_logit,1), axis=1)


        self.saver = tf.train.Saver(var_list=tf.global_variables())
        config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess  = tf.Session(config = config)
        # logs_path="./data"
        # self.train_writer = tf.summary.FileWriter( logs_path,
        #                                     self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def from_importance_weights(self,log_rhos,discounts, rewards, values, 
        bootstrap_value,clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0, 
        lambda_=1.0,name='vtrace_from_importance_weights'):


        with tf.name_scope(name):
            rhos = tf.exp(log_rhos)
            if clip_rho_threshold is not None:
                clipped_rhos = tf.minimum(clip_rho_threshold, rhos, name='clipped_rhos')
            else:
                clipped_rhos = rhos

            cs = tf.minimum(1.0, rhos, name='cs')
            cs *= tf.convert_to_tensor(lambda_, dtype=tf.float32)

            # Append bootstrapped value to get [v1, ..., v_t+1]
            values_t_plus_1 = tf.concat(
                [values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
            deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

            sequences = (discounts, cs, deltas)

            def scanfunc(acc, sequence_item):
                discount_t, c_t, delta_t = sequence_item
                return delta_t + discount_t * c_t * acc

            initial_values = tf.zeros_like(bootstrap_value)
            vs_minus_v_xs = tf.scan(
                fn=scanfunc,
                elems=sequences,
                initializer=initial_values,
                parallel_iterations=1,
                back_prop=False,
                reverse=True,
                name='scan')
            vs = tf.add(vs_minus_v_xs, values)
            self.c1 = deltas
            
            self.c3 = vs_minus_v_xs
            self.c4 = bootstrap_value
            self.c5 = values



            # Advantage for policy gradient.
            vs_t_plus_1 = tf.concat([
                vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
            if clip_pg_rho_threshold is not None:
                clipped_pg_rhos = tf.minimum(clip_pg_rho_threshold, rhos,
                                            name='clipped_pg_rhos')
            else:
                clipped_pg_rhos = rhos
            pg_advantages = (
                clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values))
            print('pg_advantages', pg_advantages.shape)
            self.c2 = pg_advantages
            # Make sure no gradients backpropagated through the returned values.
            return tf.stop_gradient(vs),tf.stop_gradient(pg_advantages)

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expend_g = tf.expand_dims(g, 0)
                grads.append(expend_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def cat_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1), z0

    # def get_params(self):
    #     pi_params, c_params = self.sess.run([self.pi_params, self.c_params])
    #     return [pi_params, c_params] 

    # def pull_params(self, pi_params, c_params):
    #     self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.pi_params, pi_params)]
    #     self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, c_params)]
    #     self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def actor_learn(self, image_ph, obs1_ph, obs2_ph,\
                    act_ph, rew_ph, done_ph, log_old_pi_ph):
        self.sess.run(self.a_train,{self.log_old_pi_ph:log_old_pi_ph, self.act_ph:act_ph, self.rew_ph:rew_ph, self.done_ph:done_ph, self.obs1_ph:obs1_ph, \
                                    self.obs2_ph:obs2_ph, self.image_ph:image_ph,\
                                    self.eval_bn_flag:True, self.targ_bn_flag:True})

    def critic_learn(self, image_ph, obs1_ph, obs2_ph,\
                    act_ph, rew_ph, done_ph, log_old_pi_ph):
        self.sess.run(self.c_train,{self.log_old_pi_ph:log_old_pi_ph, self.act_ph:act_ph, self.rew_ph:rew_ph, self.done_ph:done_ph, self.obs1_ph:obs1_ph, \
                                    self.obs2_ph:obs2_ph, self.image_ph:image_ph,\
                                    self.eval_bn_flag:True, self.targ_bn_flag:True})
       
    def save_tensor(self,s,a,adv_ph,ret_ph,episode):
        summary=self.sess.run(self.merge_loss,{self.a:a,self.adv_ph:adv_ph,self.ret_ph:ret_ph,self.S:s, self.eval_bn_flag:True, self.targ_bn_flag:True, self.critic_bn_flag:True})
        self.train_writer.add_summary(summary, episode)

    def save_ep_reward(self, ep_reward, episode):
        summary = self.sess.run(self.merged_reward, {self.ep_reward:ep_reward})
        self.train_writer.add_summary(summary, episode)

    def get_train_op(self, optimizer, params, loss):
        max_grad_norm = 40
        grads_and_var = optimizer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        op = optimizer.apply_gradients(grads_and_var)
        return op
        
    def check_params(self, image_ph, obs1_ph, obs2_ph,\
                    act_ph, rew_ph, done_ph, log_old_pi_ph):
        c1 = self.sess.run(self.c1,{self.log_old_pi_ph:log_old_pi_ph, self.act_ph:act_ph, self.rew_ph:rew_ph, self.done_ph:done_ph, self.obs1_ph:obs1_ph, \
                                    self.image_ph:image_ph,\
                                    self.critic_bn_flag:True})
        c2 = self.sess.run(self.c2,{self.log_old_pi_ph:log_old_pi_ph, self.act_ph:act_ph, self.rew_ph:rew_ph, self.done_ph:done_ph, self.obs1_ph:obs1_ph, \
                                    self.image_ph:image_ph,\
                                    self.critic_bn_flag:True})
        c3 = self.sess.run(self.c3,{self.log_old_pi_ph:log_old_pi_ph, self.act_ph:act_ph, self.rew_ph:rew_ph, self.done_ph:done_ph, self.obs1_ph:obs1_ph, \
                                    self.image_ph:image_ph,\
                                    self.critic_bn_flag:True})
        c4 = self.sess.run(self.c4,{self.log_old_pi_ph:log_old_pi_ph, self.act_ph:act_ph, self.rew_ph:rew_ph, self.done_ph:done_ph, self.obs1_ph:obs1_ph, \
                                    self.image_ph:image_ph,\
                                    self.critic_bn_flag:True})
        c5 = self.sess.run(self.c5,{self.log_old_pi_ph:log_old_pi_ph, self.act_ph:act_ph, self.rew_ph:rew_ph, self.done_ph:done_ph, self.obs1_ph:obs1_ph, \
                                    self.image_ph:image_ph,\
                                    self.critic_bn_flag:True})
        print('pg_advantages', c2)
        print('values', c5)
        print('bootstrap_value', c4)
        print('deltas', c1)
        print('vs_minus_v_xs', c3)

    #返回一个list
    def get_action(self, infer_image, infer_obs1, infer_obs2):
        log_prob = self.sess.run(self.infer_log_prob,{self.infer_image:infer_image, self.infer_obs1:infer_obs1, self.infer_obs2:infer_obs2, self.eval_bn_flag:False})
        print('log_prob', log_prob)
        return self.sess.run(self.sample_action,{self.infer_image:infer_image, self.infer_obs1:infer_obs1, self.infer_obs2:infer_obs2, self.eval_bn_flag:False})

    def get_value(self,infer_image, infer_obs1, infer_obs2):
        return self.sess.run(self.critic,{self.infer_image:infer_image, self.infer_obs1:infer_obs1, self.infer_obs2:infer_obs2, \
                                          self.critic_bn_flag:False})
    
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
        pi_params  = self.sess.run(self.pi_params)
        print('train_model_pi_params', pi_params[-1])
            
    def restore_model(self, model_path):
            self.saver.restore(self.sess, model_path)

    def conv2d(self, x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None,
            summary_tag=None):
        def intprod(x):
            return int(np.prod(x))
        with tf.variable_scope(name):
            stride_shape = [1, stride[0], stride[1], 1]
            filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = intprod(filter_shape[:3])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = intprod(filter_shape[:2]) * num_filters
            # initialize weights with random weights
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            w = tf.get_variable("W", filter_shape, dtype, tf.orthogonal_initializer(),
                                collections=collections)
            b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.orthogonal_initializer(),
                                collections=collections)

            if summary_tag is not None:
                tf.summary.image(summary_tag,
                                tf.transpose(tf.reshape(w, [filter_size[0], filter_size[1], -1, 1]),
                                            [2, 0, 1, 3]))

            return tf.nn.conv2d(x, w, stride_shape, pad) + b

    def _build_cnn(self, image, scope, bn_flag, trainable):
        with tf.variable_scope(scope):
            x = self.conv2d(image, 32, "l1", [5, 5], [3, 3], pad="VALID")
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = self.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID")
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = self.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID")
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.Flatten()(x) 
            return x

    def _build_a(self, cnn_out, obs1, obs2, scope, bn_flag, trainable, num_trajectories, is_infer):
        init_w = tf.random_normal_initializer(0., 0.01)
        init_b = tf.constant_initializer(0.1)
        with tf.variable_scope(scope):
            if self._use_image:
                x = cnn_out
                #x = tf.concat([obs1, cnn_out], axis = -1)
            else:
                x = obs1
            n_l1  = 256
            w1_s1 = tf.get_variable('w1_s1', [x.shape[-1], n_l1], initializer=init_w, trainable=trainable)
            #w1_s2 = tf.get_variable('w1_s2', [obs2.shape[-1], n_l1], initializer=init_w, trainable=trainable)
            b1   = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w1_s1) +  b1# tf.matmul(obs2, w1_s2) +
            x    = tf.nn.leaky_relu(x, alpha=0.2)
            
            n_l2 = self.a_dim
            w2_s = tf.get_variable('w2_s', [x.shape[-1], n_l2], initializer=init_w, trainable=trainable)
            b2   = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w2_s) + b2
            if is_infer:
                return x
            else:
                return tf.reshape(x, [-1, num_trajectories, self.a_dim])
        
    def _build_c(self, cnn_out, obs1, obs2, scope, bn_flag, trainable, num_trajectories, is_infer):
        init_w = tf.random_normal_initializer(0., 0.01)
        init_b = tf.constant_initializer(0.1)
        with tf.variable_scope(scope):

            if self._use_image:
                x = cnn_out
                #x = tf.concat([obs1, cnn_out], axis = -1)
            else:
                x = obs1
            n_l1  = 256
            w1_s1 = tf.get_variable('w1_s1', [x.shape[-1], n_l1], initializer=init_w, trainable=trainable)
            #w1_s2 = tf.get_variable('w1_s2', [obs2.shape[-1], n_l1], initializer=init_w, trainable=trainable)
            b1   = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w1_s1) +  b1# tf.matmul(obs2, w1_s2) +
            x    = tf.nn.leaky_relu(x, alpha=0.2)
            n_l2 = 1
            w2_s = tf.get_variable('w2_s', [x.shape[-1], n_l2], initializer=init_w, trainable=trainable)
            b2   = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w2_s) + b2
            x    = tf.squeeze(x)
            if is_infer:
                return x
            else:
                return tf.reshape(x, [-1, num_trajectories])
        


if __name__ == '__main__':
    agent = build_agent_gpu(image_shape = (4, 4), a_dim = 4, s1_dim = 20, s2_dim = 10, num_gpus = 1, using_image = False, num_trajectories = 2)
    image_ph = next_image_ph = np.zeros((4, 2, 4, 4, 1))
    obs1_ph = next_obs1_ph = np.ones((4, 2, 20)) 
    obs2_ph = next_obs2_ph = np.ones((4, 2, 10)) 
    a_ph = [[1, 3],[1, 3],[1, 3],[1, 3] ]
    rew_ph = [[-1, 10], [-1, 10], [-1, 10], [-1, 10]]
    done_ph = np.zeros((4, 2)) 
    log_old_pi_ph = [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]
    for i in range(30):
        agent.critic_learn(image_ph, next_image_ph, obs1_ph, next_obs1_ph, obs2_ph, next_obs2_ph,\
                        a_ph, rew_ph, done_ph, log_old_pi_ph)
        agent.actor_learn(image_ph, next_image_ph, obs1_ph, next_obs1_ph, obs2_ph, next_obs2_ph,\
                        a_ph, rew_ph, done_ph, log_old_pi_ph)

        infer_image = np.zeros((1, 4, 4, 1))
        infer_obs1 = np.ones((1,20))
        infer_obs2 = np.ones((1,10))
        agent.get_action(infer_image, infer_obs1, infer_obs2)
    agent.check_params(image_ph, next_image_ph, obs1_ph, next_obs1_ph, obs2_ph, next_obs2_ph,\
                        a_ph, rew_ph, done_ph, log_old_pi_ph)