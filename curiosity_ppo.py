# encoding: utf-8
import tensorflow as tf
import numpy as np
import math
import os

from sampler_files.tf_utils import get_logp_action_mask, cat_entropy_action_mask, get_sample_op_action_mask, conv2d
from sampler_files.config import learning_rate_config, ppo_config
from sampler_files.build_predictor import build_state_action_predictor

class build_agent_gpu():
    def __init__(self, image_shape = (72, 48, 3), a_dim = 10, s1_dim = 30, s2_dim = 10, single_gpu_batch_size = 64, gpu_list = [1], using_image = True, sess = None):

        self._use_image = using_image
        self.a_dim      = a_dim
        visible_gpu = ','.join([str(x) for x in gpu_list])
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpu
        with tf.device('/cpu:0'):
            self.image      = tf.placeholder(tf.float32, [None, *image_shape], name = 'image')
            self.next_image = tf.placeholder(tf.float32, [None, *image_shape], name = 'next_image')
            self.obs1       = tf.placeholder(tf.float32, [None, s1_dim], name = 'obs1')
            self.next_obs1  = tf.placeholder(tf.float32, [None, s1_dim], name = 'next_obs1')
            self.obs2       = tf.placeholder(tf.float32, [None, s2_dim], name = 'obs2')
            self.act        = tf.placeholder(tf.int32,(None,), name = 'act')
            self.adv_ph     = tf.placeholder(tf.float32,(None,), name = 'adv_ph')
            self.ret_ph     = tf.placeholder(tf.float32,(None,), name = 'ret_ph')

            tf.add_to_collection('ph', self.image)
            tf.add_to_collection('ph', self.obs1)
            tf.add_to_collection('ph', self.obs2)
            self.ep_reward = tf.placeholder(tf.float32)
            self.eval_bn_flag = tf.placeholder(tf.bool)
            self.targ_bn_flag = tf.placeholder(tf.bool)
            
            kl = []
            grads_a = []
            grads_c = []
            grads_pred = []
            v_losses = []
            pi_losses = []
            pred_losses = []
            self.check_list =[]
            self.lr_a = tf.Variable(learning_rate_config.start_lr)
            self.lr_c = tf.Variable(learning_rate_config.start_lr)
            decay = tf.constant(learning_rate_config.lr_decay_rate)
            self.lr_a_op = tf.assign(self.lr_a, tf.multiply(self.lr_a, decay))
            self.lr_c_op = tf.assign(self.lr_c, tf.multiply(self.lr_c, decay))

            self.lr_a_init_op = tf.assign(self.lr_a, learning_rate_config.restart_lr)
            self.lr_c_init_op = tf.assign(self.lr_c, learning_rate_config.restart_lr)

            self.a_optimizer = tf.train.AdamOptimizer(self.lr_a)
            self.c_optimizer = tf.train.AdamOptimizer(self.lr_c)
            self.pred_optimizer = tf.train.AdamOptimizer(self.lr_c)
            for i in range(len(gpu_list)):
                single_gpu_image  = self.image[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                single_gpu_next_image  = self.next_image[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                single_gpu_obs1   = self.obs1[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                single_gpu_obs2   = self.obs2[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                single_gpu_act    = self.act[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                single_gpu_adv    = self.adv_ph[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                single_gpu_ret    = self.ret_ph[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                single_gpu_obs1   = tf.clip_by_value(single_gpu_obs1, -3, 3)
                single_gpu_adv    = tf.clip_by_value(single_gpu_adv, -20, 20)
            

                intrinsic_reward, predictor_loss, invloss, forwardloss = build_state_action_predictor(
                    single_gpu_image, single_gpu_next_image, single_gpu_act, a_dim, 
                    self._build_cnn, self.eval_bn_flag,
                    embedding_size = 256, scope = 'predictor'
                    )


                with tf.variable_scope('CNN', reuse=tf.AUTO_REUSE):
                    critic_cnn  = self._build_cnn(single_gpu_image ,'critic', self.eval_bn_flag, True)    
                    actor_cnn   = self._build_cnn(single_gpu_image ,'actor', self.eval_bn_flag, True)  

                with tf.variable_scope('Critic', reuse=tf.AUTO_REUSE):
                    critic    = tf.squeeze(self._build_c(critic_cnn, single_gpu_obs1, single_gpu_obs2,\
                                                'pi', self.eval_bn_flag, True), axis=1)

                
                with tf.variable_scope('Actor', reuse=tf.AUTO_REUSE):
                    logit       = self._build_a(actor_cnn, single_gpu_obs1, single_gpu_obs2, 'pi', self.eval_bn_flag, True)
                    log_logit   = self._build_a(actor_cnn, single_gpu_obs1, single_gpu_obs2, 'old_pi', self.targ_bn_flag, False)
    
                log_pi     = get_logp_action_mask(single_gpu_act, logits = logit, action_masks = single_gpu_obs2)
                log_old_pi = get_logp_action_mask(single_gpu_act, logits = log_logit, action_masks = single_gpu_obs2)
                log_old_pi = tf.stop_gradient(log_old_pi)

                ratio = tf.exp(log_pi - log_old_pi)
                ratio = tf.clip_by_value(ratio, 0, 3)
                self.check_list.append(ratio)
                kl.append(log_old_pi-log_pi)
                min_adv = tf.where(single_gpu_adv>0, (1+ppo_config.clip_ratio)*single_gpu_adv, (1-ppo_config.clip_ratio)*single_gpu_adv)

                entropy = cat_entropy_action_mask(logit, action_masks = single_gpu_obs2)
                self.check_list.append(entropy)
                pi_loss = -tf.reduce_mean(tf.minimum(ratio * single_gpu_adv, min_adv))-tf.reduce_mean(entropy * ppo_config.entropy_rate)
                self.check_list.append(pi_loss)
                v_loss  = tf.reduce_mean((single_gpu_ret - critic)**2*0.5)
                self.check_list.append(v_loss)
                self.check_list.append(invloss)
                self.check_list.append(forwardloss)
                v_losses.append(v_loss)
                pi_losses.append(pi_loss)
                pred_losses.append(predictor_loss)


                if not using_image:
                    self.train_pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/pi')
                    self.c_params        = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic')
                    self.pi_params       = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/pi')
                    self.old_pi_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/old_pi')
                else:
                    self.train_pi_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'CNN/actor')+\
                                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Actor/pi')
                    self.c_params        = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'CNN/critic')+\
                                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Critic')
                                        
                    self.pi_params       = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/pi')
                    self.old_pi_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/old_pi')
                self.pred_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'predictor')

                with tf.device('/gpu:{}'.format(gpu_list[i])):
                    grads_pred.append(self.pred_optimizer.compute_gradients(predictor_loss, self.pred_params))
                    grads_a.append(self.a_optimizer.compute_gradients(pi_loss, self.train_pi_params))
                    #(self.get_grads(self.a_optimizer, self.pi_params, pi_loss))
                    grads_c.append(self.c_optimizer.compute_gradients(v_loss, self.c_params))
                    #(self.get_grads(self.c_optimizer, self.c_params, v_loss))
        grad_a = self.average_gradients(grads_a)
        grad_c = self.average_gradients(grads_c)
        grad_pred = self.average_gradients(grads_pred)
        grad_a = self.clip_grad(grad_a)
        grad_c = self.clip_grad(grad_c)
        grad_pred = self.clip_grad(grad_pred)
        critic_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'CNN/critic') + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'Critic') 
        print('critic_update_ops', critic_update_ops)
        actor_update_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'CNN/actor') + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'Actor/pi') 
        print('actor_update_ops', actor_update_ops)
        pred_update_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'predictor')
        print('pred_update_ops', pred_update_ops)
        with tf.control_dependencies(critic_update_ops):
            self.c_train = self.c_optimizer.apply_gradients(grad_c)
        with tf.control_dependencies(actor_update_ops):
            self.a_train = self.a_optimizer.apply_gradients(grad_a)
        with tf.control_dependencies(pred_update_ops):
            self.pred_train = self.pred_optimizer.apply_gradients(grad_pred)
        self.v_loss = self.average_loss(v_losses)
        self.pi_loss = self.average_loss(pi_losses)
        self.predictor_loss = self.average_loss(pred_losses)
        
        # v_sum_op  = tf.summary.scalar('v_loss',v_loss)
        # pi_sum_op = tf.summary.scalar('pi_loss',pi_loss)
        # self.merge_loss = tf.summary.merge([v_sum_op, pi_sum_op])
        # sum_reward = tf.summary.scalar('self.ep_reward',self.ep_reward)
        # self.merged_reward = tf.summary.merge([sum_reward])
        
        # update_ops1 = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope = 'Actor/eval_a')
        # with tf.control_dependencies(update_ops1):
        

        # update_ops2 = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope = 'Critic')
        # with tf.control_dependencies(update_ops2):
        

        ###TODO
        self.kl = tf.reduce_mean(kl)
        self.set_equal = [tf.assign(o_p, p)
                                    for p,o_p in zip(self.pi_params,self.old_pi_params)]

        self.saver = tf.train.Saver(var_list=tf.global_variables())
        self.sess  = sess
        logs_path="./data"
        self.train_writer = tf.summary.FileWriter( logs_path,
                                            self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

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

    def init_lr(self):
        self.sess.run([self.lr_a_init_op, self.lr_c_init_op])
        
    def update_lr(self):
        self.sess.run([self.lr_a_op, self.lr_c_op])

    def check_lr(self):
        lr_a = self.sess.run(self.a_optimizer ._lr)
        lr_c = self.sess.run(self.c_optimizer ._lr)
        print('Learning rate a',  lr_a)
        print('Learning rate c',  lr_c)
        return lr_a, lr_c
    # def get_params(self):
    #     pi_params, c_params = self.sess.run([self.pi_params, self.c_params])
    #     return [pi_params, c_params] 

    # def pull_params(self, pi_params, c_params):
    #     self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.pi_params, pi_params)]
    #     self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, c_params)]
    #     self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def clip_grad(self, grads_and_var):
        max_grad_norm = 40

        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        return grads_and_var

    def actor_learn(self, sampler_dict):
        image = sampler_dict['image_ph']
        obs1 = sampler_dict['obs1_ph']
        obs2 = sampler_dict['obs2_ph']
        act = sampler_dict['act_ph']
        adv_ph = sampler_dict['adv_ph']
        next_image = sampler_dict['next_image_ph']
        _, pi_loss, _, predictor_loss = self.sess.run([self.a_train, self.pi_loss, self.pred_train, self.predictor_loss],\
        {self.image:image, self.next_image:next_image, self.act:act, self.obs1:obs1, self.obs2:obs2, self.adv_ph:adv_ph, \
                                    self.eval_bn_flag:True, self.targ_bn_flag:True})
        print('predictor_loss', predictor_loss)

        return pi_loss

    def pred_learn(self, sampler_dict):
        image = sampler_dict['image_ph']
        next_image = sampler_dict['next_image_ph']
        act = sampler_dict['act_ph']
        _, predictor_loss, invloss, forwardloss = self.sess.run([self.pred_train, self.predictor_loss, self.invloss, self.forwardloss],\
        {self.image:image, self.act:act, self.next_image:next_image, \
                                    self.eval_bn_flag:True, self.targ_bn_flag:True})
        print('invloss', invloss)
        print('forwardloss', forwardloss)
        return predictor_loss

    def critic_learn(self, sampler_dict):
        image = sampler_dict['image_ph']
        obs1 = sampler_dict['obs1_ph']
        obs2 = sampler_dict['obs2_ph']
        ret_ph = sampler_dict['ret_ph']
        _, v_loss = self.sess.run([self.c_train, self.v_loss],\
        {self.ret_ph:ret_ph,self.image:image, self.obs1:obs1, self.obs2:obs2, self.eval_bn_flag:True, self.targ_bn_flag:True})
        return v_loss
       
    def save_tensor(self,s,a,adv_ph,ret_ph,episode):
        summary=self.sess.run(self.merge_loss,{self.a:a,self.adv_ph:adv_ph,self.ret_ph:ret_ph,self.S:s, self.eval_bn_flag:True, self.targ_bn_flag:True})
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
        
    def check_params(self):
        for p in self.train_pi_params:
            print(p)
            print(self.sess.run(p))
        for p in self.c_params:
            print(p)
            print(self.sess.run(p))

    def debug(self, sampler_dict, count):

        image = sampler_dict['image_ph']
        obs1 = sampler_dict['obs1_ph']
        obs2 = sampler_dict['obs2_ph']
        a = sampler_dict['act_ph']
        adv_ph = sampler_dict['adv_ph']
        ret_ph = sampler_dict['ret_ph']
        check_list = self.sess.run(self.check_list,\
            {self.ret_ph:ret_ph,self.image:image, self.obs1:obs1, self.obs2:obs2})

        ratio0 = check_list[-8]
        entropy0 = check_list[-7]
        pi_loss0 = check_list[-6]
        v_loss0 = check_list[-5]
        for single_ratio in ratio0:
             if math.isnan(single_ratio):
                print('nan in debug ratio')
                np.save(count+'imagedebug', image)
                np.save(count+'obs1debug', obs1)
                np.save(count+'obs2debug', obs2)
                np.save(count+'actdebug', a)
                np.save(count+'advdebug', adv_ph)
                np.save(count+'retdebug', ret_ph)
        if math.isnan(entropy0) or math.isnan(pi_loss0) or math.isnan(v_loss0):
            print('nan in debug')
            np.save(count+'ratio0', np.array(ratio0))
            np.save(count+'entropy0', np.array(entropy0))
            np.save(count+'pi_loss0', np.array(pi_loss0))
            np.save(count+'v_loss0', np.array(v_loss0))
            np.save(count+'imagedebug', image)
            np.save(count+'obs1debug', obs1)
            np.save(count+'obs2debug', obs2)
            np.save(count+'actdebug', a)
            np.save(count+'advdebug', adv_ph)
            np.save(count+'retdebug', ret_ph)
        ratio1 = check_list[-4]
        entropy1 = check_list[-3]
        pi_loss1 = check_list[-2]
        v_loss1 = check_list[-1]
        for single_ratio in ratio1:
             if math.isnan(single_ratio):
                print('nan in debug ratio')
                np.save(count+'imagedebug', image)
                np.save(count+'obs1debug', obs1)
                np.save(count+'obs2debug', obs2)
                np.save(count+'actdebug', a)
                np.save(count+'advdebug', adv_ph)
                np.save(count+'retdebug', ret_ph)
        if math.isnan(entropy1) or math.isnan(pi_loss1) or math.isnan(v_loss1):
            print('nan in debug')
            np.save(count+'ratio0', np.array(ratio0))
            np.save(count+'entropy0', np.array(entropy0))
            np.save(count+'pi_loss0', np.array(pi_loss0))
            np.save(count+'v_loss0', np.array(v_loss0))
            np.save(count+'imagedebug', image)
            np.save(count+'obs1debug', obs1)
            np.save(count+'obs2debug', obs2)
            np.save(count+'actdebug', a)
            np.save(count+'advdebug', adv_ph)
            np.save(count+'retdebug', ret_ph)
    
    def update_old_pi(self):#更新旧网络参数，使之与当前的actor完全相同
        self.sess.run(self.set_equal)

    def get_kl(self, image, obs1, obs2, a):
        return self.sess.run(self.kl,{self.image:image, self.obs1:obs1, self.obs2:obs2,self.a:a,\
                                      self.eval_bn_flag:True, self.targ_bn_flag:True})
    #返回一个list
    def get_action(self, image, obs1, obs2):
        return self.sess.run(self.sample_action,{self.image:image, self.obs1:obs1, self.obs2:obs2, self.eval_bn_flag:False})
        
    def get_value(self,image, obs1, obs2):
        return self.sess.run(self.critic,{self.image:image, self.obs1:obs1, self.obs2:obs2})
    
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
            
    def restore_model(self, model_path):
            self.saver.restore(self.sess, model_path)

    def _build_cnn(self, image, scope, bn, trainable):
        with tf.variable_scope(scope):
            x = conv2d(image, 16, "l1", [5, 5], [3, 3], pad="SAME")
            x = tf.layers.batch_normalization(x, training=bn)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = conv2d(x, 32, "l2", [4, 4], [2, 2], pad="SAME")
            x = tf.layers.batch_normalization(x, training=bn)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = conv2d(x, 32, "l3", [3, 3], [1, 1], pad="SAME")
            x = tf.layers.batch_normalization(x, training=bn)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.Flatten()(x) 
            return x

    def _build_a(self, cnn_out, obs1, obs2, scope, bn, trainable):
        init_w = tf.random_normal_initializer(0., 0.01)
        init_b = tf.constant_initializer(0.1)
        with tf.variable_scope(scope):

            x     = tf.concat([obs1, obs2], axis = -1)
            n_s   = 1024
            w_s   = tf.get_variable('w_s', [x.shape[-1], n_s], initializer=init_w, trainable=trainable)
            b_s   = tf.get_variable('b_s', [1, n_s], initializer=init_b, trainable=trainable)
            x     = tf.matmul(x, w_s) + b_s
            x     = tf.layers.batch_normalization(x, training=bn)
            x     = tf.nn.leaky_relu(x, alpha=0.2)
            x     = tf.concat([x, cnn_out], axis = -1)

            n_l1 = 512
            w1_s = tf.get_variable('w1_s', [x.shape[-1], n_l1], initializer=init_w, trainable=trainable)
            b1   = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w1_s) + b1
            x    = tf.layers.batch_normalization(x, training=bn)
            x    = tf.nn.leaky_relu(x, alpha=0.2)
            
            n_l2 = 256
            w2_s = tf.get_variable('w2_s', [x.shape[-1], n_l2], initializer=init_w, trainable=trainable)
            b2   = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w2_s) + b2
            x    = tf.layers.batch_normalization(x, training=bn)
            x    = tf.nn.leaky_relu(x, alpha=0.2)

            n_l3 = self.a_dim
            w3_s = tf.get_variable('w3_s', [x.shape[-1], n_l3], initializer=init_w, trainable=trainable)
            b3   = tf.get_variable('b3', [1, n_l3], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w3_s) + b3

            return x
        
    def _build_c(self, cnn_out, obs1, obs2, scope, bn, trainable):
        init_w = tf.random_normal_initializer(0., 0.01)
        init_b = tf.constant_initializer(0.1)
        with tf.variable_scope(scope):

            x     = tf.concat([obs1, obs2], axis = -1)
            n_s   = 1024
            w_s   = tf.get_variable('w_s', [x.shape[-1], n_s], initializer=init_w, trainable=trainable)
            b_s   = tf.get_variable('b_s', [1, n_s], initializer=init_b, trainable=trainable)
            x     = tf.matmul(x, w_s) + b_s
            x     = tf.layers.batch_normalization(x, training=bn)
            x     = tf.nn.leaky_relu(x, alpha=0.2)
            x     = tf.concat([x, cnn_out], axis = -1)

            n_l1 = 512
            w1_s = tf.get_variable('w1_s', [x.shape[-1], n_l1], initializer=init_w, trainable=trainable)
            b1   = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w1_s) + b1
            x    = tf.layers.batch_normalization(x, training=bn)
            x    = tf.nn.leaky_relu(x, alpha=0.2)
            
            n_l2 = 256
            w2_s = tf.get_variable('w2_s', [x.shape[-1], n_l2], initializer=init_w, trainable=trainable)
            b2   = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w2_s) + b2
            x    = tf.layers.batch_normalization(x, training=bn)
            x    = tf.nn.leaky_relu(x, alpha=0.2)

            n_l3 = 1
            w3_s = tf.get_variable('w3_s', [x.shape[-1], n_l3], initializer=init_w, trainable=trainable)
            b3   = tf.get_variable('b3', [1, n_l3], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w3_s) + b3

            return x
        


if __name__ == '__main__':
    config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess  = tf.Session(config = config)
    agent = build_agent_gpu(image_shape = (72, 48, 3), a_dim = 10, s1_dim = 30, s2_dim = 10, single_gpu_batch_size = 64, gpu_list = [1], using_image = True, sess = sess)

    sample_dict = dict()
    act_ph = np.ones(64)
    image_ph = np.ones((64, 72, 48, 3))
    next_image_ph = np.ones((64, 72, 48, 3))*2
    sample_dict['act_ph'] = act_ph
    sample_dict['image_ph'] = image_ph
    sample_dict['next_image_ph'] = next_image_ph
    for i in range(400):
        pred_loss = agent.pred_learn(sample_dict)
        print('pred_loss', pred_loss)