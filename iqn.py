# encoding: utf-8
import tensorflow as tf
import numpy as np
import math
import random
import os
TAU = 0.96

class build_agent_gpu():
    def __init__(self, a_dim, batch_size, gpu_list):
        visible_gpu = ','.join([str(x) for x in gpu_list])
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpu
        with tf.device('/cpu:0'):
            self.a_dim           = a_dim
            self.obs_ph          = tf.placeholder(tf.float32, [batch_size, 72, 48, 4])
            self.next_obs_ph     = tf.placeholder(tf.float32, [batch_size, 72, 48, 4])
            self.act_ph          = tf.placeholder(tf.int32, (batch_size,))
            self.rew_ph          = tf.placeholder(tf.float32, (batch_size,))
            self.done_ph         = tf.placeholder(tf.float32, (batch_size,))
            self.eval_bn_flag    = tf.placeholder(tf.bool)
            self.targ_bn_flag    = tf.placeholder(tf.bool)
            self.batch_size = batch_size
            self.kappa = 1
            self.num_tau_samples=32
            self.num_tau_prime_samples=32
            self.quantile_embedding_dim=64 # used for extract feature from input [0, 1] distribution tao.
            self.cumulative_gamma = 0.99 # math.pow(gamma, update_horizon)
            self.epsilon_eval = 0.001
            self.epsilon_decay_period = 250000
            self.training_steps = 0
            self.min_replay_history = 10000
            self.epsilon_train = 0.01

            self.ep_reward = tf.placeholder(tf.float32)

            sum_reward = tf.summary.scalar('ep_reward',self.ep_reward)

            self.merged_reward = tf.summary.merge([sum_reward])
            logs_path="./data"
            config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            config.gpu_options.allow_growth = True

            self.sess  = tf.Session(config = config)

            self.optimizer = tf.train.RMSPropOptimizer(
                                learning_rate=0.00025,
                                decay=0.95,
                                momentum=0.0,
                                epsilon=0.00001,
                                centered=True)

            num_device = len(gpu_list)
            single_device_obs_ph          = tf.split(self.obs_ph, num_device)
            single_device_next_obs_ph     = tf.split(self.next_obs_ph, num_device)
            single_device_act_ph          = tf.split(self.act_ph, num_device)
            single_device_rew_ph          = tf.split(self.rew_ph, num_device)
            single_device_done_ph         = tf.split(self.done_ph, num_device)
            grads = []
            losses = []
            single_gpu_batch_size = batch_size//num_device
            for i in range(len(num_device)):
                with tf.variable_scope('main', reuse=tf.AUTO_REUSE):
                    # self.execute_quantile_values,     self.execute_quantiles      = \
                    # self._build_net(self.obs_ph, 'eval', self.eval_bn_flag, True)
                    # self.execute_targ_quantile_values, self.execute_targ_quantiles  = \
                    # self._build_net(self.next_obs_ph, 'targ', self.targ_bn_flag, True)
                    eval_quantile_values, eval_quantiles  = \
                    self._build_net(single_device_obs_ph[i], 'eval', self.eval_bn_flag, True)

                    targ_quantile_values, target_targ_quantiles  = \
                    self._build_net(single_device_next_obs_ph[i], 'targ', self.targ_bn_flag, True)

                    self._q_values = tf.reduce_mean(eval_quantile_values, axis=0)
                    self._q_argmax = tf.squeeze(tf.argmax(self._q_values, axis=-1))
        
                    # 把num_tau_samples采的次数，累加求平均     
                    eval_next_quantile_values, _  = \
                    self._build_net(single_device_next_obs_ph[i], 'eval', self.eval_bn_flag, True)
                    print('eval_next_quantile_values', eval_next_quantile_values)
                    eval_next_quantile_values = tf.reshape(eval_next_quantile_values, (self.num_tau_samples, single_gpu_batch_size, a_dim))
                    eval_next_values = tf.reduce_mean(eval_next_quantile_values, axis=0)
                    # Shape: batch_size.代表每一个batch选择的最大动作，int
                    target_qt_argmax = tf.squeeze(tf.argmax(eval_next_values, axis=-1))

                    self.eval_params = tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, scope='main/eval')
                    self.targ_params = tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, scope='main/targ')   
                    loss = self._build_train_op(single_device_act_ph[i], single_device_rew_ph[i], single_device_done_ph[i], target_qt_argmax,
                                                eval_quantile_values,
                                                targ_quantile_values,
                                                eval_quantiles,
                                                target_targ_quantiles)
                    losses.append(loss)
                    with tf.device('/gpu:{}'.format(gpu_list[i])):
                        grads.append(self.optimizer.compute_gradients(loss, self.eval_params))
                    self.td_error = loss     

        grad = self.average_gradients(grads)
        grad = self.clip_grad(grad)
        self.loss = self.average_loss(losses)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope = 'main/eval')
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.apply_gradients(grad)

        self.targ_update_ops = [tf.assign(t, (1 - TAU) * e + TAU *t)
                                            for e,t in zip(self.eval_params, self.targ_params)]

        self.saver = tf.train.Saver(var_list=tf.global_variables())
        self.summary_writer = tf.summary.FileWriter( logs_path,
                                            self.sess.graph)
        self.sess.run(tf.global_variables_initializer())


    def get_params(self):
        eval_params, targ_params = self.sess.run([self.eval_params, self.targ_params])
        return [eval_params, targ_params] 

    def pull_params(self, eval_params, targ_params):
        self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.eval_params, eval_params)]
        self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.targ_params, targ_params)]
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def _build_target_quantile_values_op(self, single_device_rew_ph, single_device_done_ph, target_qt_argmax, targ_quantile_values):
        """Build an op used as a target for return values at given quantiles.

        Returns:
        An op calculating the target quantile return.
        """

        batch_size = single_device_rew_ph.shape[-1]
        rew_tiled = tf.reshape(tf.tile(single_device_rew_ph, [self.num_tau_prime_samples]), 
                                    [self.num_tau_prime_samples, -1])

        is_terminal_multiplier = 1. - tf.to_float(single_device_done_ph)
        # Incorporate terminal state to discount factor.
        # size of gamma_with_terminal: (num_tau_prime_samples x batch_size) x 1.
        gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
        gamma_with_terminal = tf.reshape(tf.tile(gamma_with_terminal, [self.num_tau_prime_samples]), 
                                    [self.num_tau_prime_samples, -1])
        # Get the indices of the maximium Q-value across the action dimension.
        # Shape of next_qt_argmax: (num_tau_prime_samples x batch_size) x 1.
        
        # shape num_tau_prime_samples x batch_size， 代表每一个batch中target网络选择的最大动作，第一维度全部重复 tudo
        next_qt_argmax = tf.reshape(tf.tile(target_qt_argmax, [self.num_tau_prime_samples]), 
                                            [self.num_tau_prime_samples, -1])

        targ_quantile_values = tf.reshape(targ_quantile_values, (self.num_tau_prime_samples, batch_size, self.a_dim))
        target_quantile_values = tf.reduce_sum(tf.one_hot(next_qt_argmax, self.a_dim)
                                        * targ_quantile_values, axis=-1, keepdims=False)

        return rew_tiled + gamma_with_terminal * target_quantile_values

    def _build_train_op(self, 
                        single_device_act_ph, 
                        single_device_rew_ph, 
                        single_device_done_ph, 
                        target_qt_argmax, 
                        eval_quantile_values, 
                        targ_quantile_values,
                        eval_quantiles,
                        target_targ_quantiles):
        batch_size = single_device_act_ph.shape[-1]
        target_quantile_values = tf.stop_gradient(self._build_target_quantile_values_op(single_device_rew_ph, single_device_done_ph, target_qt_argmax, targ_quantile_values))
        self.c2 = target_quantile_values
        # Reshape to self.num_tau_prime_samples x batch_size x 1 since this is
        # the manner in which the target_quantile_values are tiled.
        # 这个值，也是随机采样的值！
        target_quantile_values = tf.reshape(target_quantile_values,
                                            [self.num_tau_prime_samples,
                                            batch_size, 1])
        target_quantile_values = tf.transpose(target_quantile_values, [1, 0, 2])
        # Transpose dimensions so that the dimensionality is batch_size x
        # self.num_tau_prime_samples x 1 to prepare for computation of
        # Bellman errors.

        # Expand the dimension by one so that it can be used to index into all the
        # quantiles when using the tf.gather_nd function (see below).
        # [ batch_size, 1]
        action_tiled = tf.reshape(tf.tile(single_device_act_ph, [self.num_tau_samples]), 
                                                [self.num_tau_samples, -1])

        eval_quantile_values = tf.reshape(eval_quantile_values, (self.num_tau_samples, batch_size, self.a_dim))
        chosen_action_quantile_values = tf.reduce_sum(tf.one_hot(action_tiled, self.a_dim)
                                        * eval_quantile_values, axis=-1, keepdims=True)
        #####  chosen_action_quantile_values 是eval网络，采样时选择的action的Q值， 拓展num_tau_samples后，每一维值不相同！！！
        # Reshape to self.num_tau_samples x batch_size x 1 since this is the manner
        # in which the quantile values are tiled.
        chosen_action_quantile_values = tf.reshape(chosen_action_quantile_values,
                                                [self.num_tau_samples,
                                                    batch_size, 1])
        chosen_action_quantile_values = tf.transpose(chosen_action_quantile_values, [1, 0, 2])
        # Transpose dimensions so that the dimensionality is batch_size x
        # self.num_tau_samples x 1 to prepare for computation of
        # Bellman errors.

        # Shape of bellman_erors and huber_loss:
        # batch_size x num_tau_prime_samples x num_tau_samples x 1.
        ### target_quantile_values.shape = batch_size x num_tau_prime_samples x 1 x1 
        ### chosen_action_quantile_values.shape = batch_size x 1 x num_tau_samples x 1

        bellman_errors = target_quantile_values[
            :, :, None, :] - chosen_action_quantile_values[:, None, :, :]
        # The huber loss (see Section 2.3 of the paper) is defined via two cases:
        # case_one: |bellman_errors| <= kappa
        # case_two: |bellman_errors| > kappa
        huber_loss_case_one = tf.to_float(
            tf.abs(bellman_errors) <= self.kappa) * 0.5 * bellman_errors ** 2
        huber_loss_case_two = tf.to_float(
            tf.abs(bellman_errors) > self.kappa) * self.kappa * (
                tf.abs(bellman_errors) - 0.5 * self.kappa)
        huber_loss = huber_loss_case_one + huber_loss_case_two

        # Reshape quantiles to batch_size x num_tau_samples x 1
        ## eval_quantiles,1之间的随机分布
        eval_quantiles = tf.reshape(
            eval_quantiles, [self.num_tau_samples, batch_size, 1])
        eval_quantiles = tf.transpose(eval_quantiles, [1, 0, 2])

        eval_quantiles = tf.to_float(tf.tile(
            eval_quantiles[:, None, :, :], [1, self.num_tau_prime_samples, 1, 1]))
        quantile_huber_loss = (tf.abs(eval_quantiles - tf.stop_gradient(
            tf.to_float(bellman_errors < 0))) * huber_loss) / self.kappa
        loss = tf.reduce_sum(quantile_huber_loss, axis=2)
        # Shape: batch_size x 1.
        loss = tf.reduce_mean(loss, axis=1)
        return loss

    def linearly_decaying_epsilon(self, decay_period, step, warmup_steps, epsilon):
        """Returns the current epsilon for the agent's epsilon-greedy policy.

        This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
        al., 2015). The schedule is as follows:
            Begin at 1. until warmup_steps steps have been taken; then
            Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
            Use epsilon from there on.

        Args:
            decay_period: float, the period over which epsilon is decayed.
            step: int, the number of training steps completed so far.
            warmup_steps: int, the number of steps taken before epsilon is decayed.
            epsilon: float, the final value to which to decay the epsilon parameter.

        Returns:
            A float, the current epsilon value computed according to the schedule.
        """
        steps_left = decay_period + warmup_steps - step
        bonus = (1.0 - epsilon) * steps_left / decay_period
        bonus = np.clip(bonus, 0., 1. - epsilon)
        return epsilon + bonus

    def train(self, obs_ph, next_obs_ph, act_ph, rew_ph, done_ph):
        self.sess.run(self.train_op,{self.act_ph:act_ph,\
                    self.rew_ph:rew_ph, self.done_ph:done_ph,\
                    self.obs_ph:obs_ph, self.next_obs_ph:next_obs_ph,\
                    self.eval_bn_flag:True, self.targ_bn_flag:True,\
                    self.obs_ph:obs_ph, self.next_obs_ph:next_obs_ph}) 
        # c2 = self.sess.run(self.loss,{self.act_replay:act_replay,\
        #             self.rew_replay:rew_replay, self.done_replay:done_replay,\
        #             self.obs_replay:obs_replay, self.next_obs_replay:next_obs_replay}) 
        # c4 = self.sess.run(self.eval_quantile_values,{self.act_replay:act_replay,\
        #             self.rew_replay:rew_replay, self.done_replay:done_replay,\
        #             self.obs_replay:obs_replay, self.next_obs_replay:next_obs_replay}) 
        # c5 = self.sess.run(self.c5,{self.act_replay:act_replay,\
        #             self.rew_replay:rew_replay, self.done_replay:done_replay,\
        #             self.obs_replay:obs_replay, self.next_obs_replay:next_obs_replay})  
        # c6 = self.sess.run(self.c7,{self.act_replay:act_replay,\
        #             self.rew_replay:rew_replay, self.done_replay:done_replay,\
        #             self.obs_replay:obs_replay, self.next_obs_replay:next_obs_replay})  
        # c1 = self.sess.run(self.targ_quantile_values,{self.act_replay:act_replay,\
        #             self.rew_replay:rew_replay, self.done_replay:done_replay,\
        #             self.obs_replay:obs_replay, self.next_obs_replay:next_obs_replay})   
        # temp_done = np.zeros_like(done_replay)
        # c2 = self.sess.run(self.loss,{self.act_replay:act_replay,\
        #             self.rew_replay:rew_replay, self.done_replay:temp_done,\
        #             self.obs_replay:obs_replay, self.next_obs_replay:next_obs_replay})            
        #c3 = self.sess.run(self.c3)
        #print('--------------------c4=', c4)
        # print('--------------------c51=', c5[0])
        # print('--------------------c52=', c5[128])
        # print('--------------------c53=', c5[256])
        # print('--------------------c54=', c5[1024])

        # print('--------------------c61=', c6[0])
        # print('--------------------c62=', c6[128])
        # print('--------------------c63=', c6[256])
        # print('--------------------c64=', c6[1024])
            #print('---------------------c2', c2)
        #print()
    def soft_update(self):
        self.sess.run(self.targ_update_ops)
    
    def save_ep_reward(self, ep_reward, episode):
        summary = self.sess.run(self.merged_reward, {self.ep_reward:ep_reward})
        self.summary_writer.add_summary(summary, episode)
    
    def get_td_error(self, obs_ph, act_ph, rew_ph, done_ph, next_obs_ph):
        return self.sess.run(self.td_error,{self.obs_ph:obs_ph, self.act_ph:act_ph, \
                                            self.rew_ph:rew_ph, self.done_ph:done_ph, \
                                            self.next_obs_ph:next_obs_ph,\
                                            self.eval_bn_flag:False, self.targ_bn_flag:False})

    def get_action(self, obs_ph, is_eval):
        if is_eval:
            epsilon = self.epsilon_eval
        else:
            epsilon = self.linearly_decaying_epsilon(
                self.epsilon_decay_period,
                self.training_steps,
                self.min_replay_history,
                self.epsilon_train)
            if self.training_steps%20000 == 0 and self.training_steps>10000:
                print('----------epsilon-------------------------', epsilon)
        if random.random() <= epsilon:
        # Choose a random action with probability epsilon.
            return random.randint(0, self.a_dim - 1)
        else:
        # Choose the action with highest Q-value at the current state.
            return self.sess.run(self._q_argmax, {self.obs_ph: obs_ph, self.eval_bn_flag:False, self.targ_bn_flag:False})
    
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
            
    def restore_model(self, model_path):
            self.saver.restore(self.sess, model_path)

    def average_loss(self, tower_loss):
        loss = tf.reduce_mean(tower_loss, 0)
        return loss

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

    def clip_grad(self, grads_and_var):
        max_grad_norm = 40

        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        return grads_and_var
        
    def _build_net(self, s, scope, bn_flag, trainable):
        
        batch_size = s.get_shape().as_list()[0]
        init_w = tf.orthogonal_initializer()
        init_b = tf.orthogonal_initializer()
        with tf.variable_scope(scope):
            
            x = self.conv2d(s, 32, "l1", [7, 7], [5, 5], pad="VALID")
            x = tf.layers.batch_normalization(x, training = bn_flag)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = self.conv2d(x, 64, "l2", [3, 3], [1, 1], pad="VALID")
            x = tf.layers.batch_normalization(x, training = bn_flag)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = self.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="SAME")
            x = tf.layers.batch_normalization(x, training = bn_flag)
            x = tf.layers.Flatten()(x) 

            state_vector_length = x.get_shape().as_list()[-1]
            state_net_tiled = tf.tile(x, [self.num_tau_samples, 1])
            quantiles_shape = [self.num_tau_samples * batch_size, 1] 

            quantiles = tf.random_uniform(
                quantiles_shape, minval=0, maxval=1, dtype=tf.float32) 
            quantile_net = tf.tile(quantiles, [1, self.quantile_embedding_dim])
            pi = tf.constant(math.pi)
            quantile_net = tf.cast(tf.range(
                1, self.quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
            quantile_net = tf.cos(quantile_net)

            n_embed = state_vector_length
            w_s = tf.get_variable('w_s', [quantile_net.shape[-1], n_embed], initializer=init_w, trainable=trainable)
            b   = tf.get_variable('b', [1, n_embed], initializer=init_b, trainable=trainable)
            quantile_net    = tf.matmul(quantile_net, w_s) + b
            quantile_net    = tf.nn.leaky_relu(quantile_net, alpha=0.2)

            x = tf.multiply(state_net_tiled, quantile_net)
            print('xshape', x)
            # if scope == 'eval' and batch_size == 64:
            #     self.c6 = x

            n_l1 = 128
            w1_s = tf.get_variable('w1_s', [x.shape[-1], n_l1], initializer=init_w, trainable=trainable)
            b1   = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w1_s) + b1
            x    = tf.nn.leaky_relu(x, alpha=0.2)
            
            n_l2 = self.a_dim
            w2_s = tf.get_variable('w2_s', [n_l1, n_l2], initializer=init_w, trainable=trainable)
            b2   = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
            quantile_values = tf.matmul(x, w2_s) + b2

            return quantile_values, quantiles
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
            w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                                collections=collections)
            b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.zeros_initializer(),
                                collections=collections)

            if summary_tag is not None:
                tf.summary.image(summary_tag,
                                tf.transpose(tf.reshape(w, [filter_size[0], filter_size[1], -1, 1]),
                                            [2, 0, 1, 3]))
            return tf.nn.conv2d(x, w, stride_shape, pad) + b

if __name__ == '__main__':

    agent = build_agent_gpu(2, 128, [2, 3])
    # print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    # print(agent.execute_quantile_values)
    # a = [1 for i in range(64)]
    # s = np.ones((64, 3))
    # s_ = np.ones((64,3))*2
    # r = -np.ones(64)
    # done = np.zeros(64)
    # for i in range(500):
    #     agent.train(s, s_, a, r, done)
    #     aaa = agent.get_action([[1,1 ,1]], True)
    #     q = agent.check_value([[1,1,1]])
        #print('qqq',q)