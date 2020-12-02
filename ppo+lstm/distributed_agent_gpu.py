# encoding: utf-8
import tensorflow as tf
import numpy as np
import os
import math

from sampler_files.tf_utils import get_logp_action_mask, cat_entropy_action_mask, get_sample_op_action_mask, conv2d, batch_norm
from BuildLstmCell import dynamic_lstm, DropoutLSTMCell, LSTMCell
from sampler_files.config import learning_rate_config, experiment_config
TAU=0.96
clip_ratio=0.2
ent_rate = 0.01
lstm_dim = experiment_config.lstm_dim
traj_length = experiment_config.traj_length
hidden_dim = experiment_config.hidden_dim
weight_decay = experiment_config.weight_decay
use_bn = experiment_config.use_bn
use_dropout = experiment_config.use_dropout
if use_dropout:
    build_lstm_cell = DropoutLSTMCell
else:
    build_lstm_cell = LSTMCell
class build_agent_gpu():
    def __init__(self, image_shape = (72, 48, 3), a_dim = 10, s1_dim = 150, s2_dim = 10, single_gpu_batch_size = 64, gpu_list = [7, 8], using_image = True, mouse_num = 2):

        self._use_image = using_image
        self.a_dim      = a_dim
        self.lstm_dim  = lstm_dim
        visible_gpu = ','.join([str(x) for x in gpu_list])
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpu
        with tf.device('/cpu:0'):
            self.image      = tf.placeholder(tf.float32, [None, traj_length, *image_shape], name = 'image')
            self.obs1       = tf.placeholder(tf.float32, [None, traj_length, s1_dim], name = 'obs1')
            self.obs2       = tf.placeholder(tf.float32, [None, traj_length, s2_dim], name = 'obs2')
            self.act        = tf.placeholder(tf.int32,[None, traj_length], name = 'act')
            self.adv_ph     = tf.placeholder(tf.float32,[None, traj_length], name = 'adv_ph')
            self.ret_ph     = tf.placeholder(tf.float32,[None, traj_length], name = 'ret_ph')
            self.new_traj_start       = tf.placeholder(tf.float32,[None, traj_length], name = 'new_traj_start')
            self.actions    = tf.placeholder(tf.float32,[None, traj_length, mouse_num * a_dim], name = 'actions')
            self.lstm_init_state = tf.placeholder(tf.float32,[None, lstm_dim*2], name = 'lstm_init_state')


            tf.add_to_collection('ph', self.image)
            tf.add_to_collection('ph', self.obs1)
            tf.add_to_collection('ph', self.obs2)
            tf.add_to_collection('ph', self.actions)
            self.ep_reward = tf.placeholder(tf.float32)
            self.eval_bn_flag = tf.placeholder(tf.bool)
            self.targ_bn_flag = tf.placeholder(tf.bool)
            self.keep_prob = tf.placeholder(tf.float32) 
            
            self.check_list = list()
            self.rnn_value_check_list = list()
            kl = []
            grads = []
            losses = []

            self.lr = tf.Variable(learning_rate_config.start_lr)
            decay = tf.constant(0.95)
            self.lr_op = tf.assign(self.lr, tf.multiply(self.lr, decay))
            self.lr_init_op = tf.assign(self.lr, learning_rate_config.restart_lr)

            # self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.optimizer = tf.train.AdamOptimizer(self.lr)

            for i in range(len(gpu_list)):
                single_gpu_image   = self.image[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                single_gpu_obs1   = self.obs1[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                single_gpu_obs2   = self.obs2[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                single_gpu_act   = self.act[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                single_gpu_actions  = self.actions[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                single_gpu_adv = self.adv_ph[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                single_gpu_ret = self.ret_ph[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                single_gpu_new_traj_start = self.new_traj_start[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                single_gpu_lstm_init_state = self.lstm_init_state[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                single_gpu_obs1 = tf.clip_by_value(single_gpu_obs1, -3, 3)
                single_gpu_adv  = tf.clip_by_value(single_gpu_adv, -20, 20)
            
                single_gpu_image = tf.reshape(single_gpu_image, (single_gpu_batch_size*traj_length, *image_shape))
                single_gpu_obs1 = tf.reshape(single_gpu_obs1, (single_gpu_batch_size*traj_length, s1_dim))
                single_gpu_obs2 = tf.reshape(single_gpu_obs2, (single_gpu_batch_size*traj_length, s2_dim))
                with tf.variable_scope('Common', reuse=tf.AUTO_REUSE):
                    cnn_out      = self._build_cnn(single_gpu_image ,'cnn', self.eval_bn_flag, True)    
                    hidden_state = self._build_hidden_state(cnn_out, single_gpu_obs1, single_gpu_obs2, 'hidden', self.eval_bn_flag, self.keep_prob, True)
                    hidden_state = tf.reshape(hidden_state, (single_gpu_batch_size, traj_length, hidden_state.shape[-1]))
                    lstm_cell = build_lstm_cell(lstm_dim, 'lstm', self.keep_prob)
                    lstm_out, lstm_hidden_state, check_values = dynamic_lstm(lstm_cell, hidden_state, single_gpu_lstm_init_state, single_gpu_new_traj_start)
                    lstm_out = tf.reshape(lstm_out, (single_gpu_batch_size*traj_length, lstm_dim))

                critic  = tf.squeeze(self._build_c(lstm_out,\
                                            single_gpu_actions, 'critic', self.eval_bn_flag, self.keep_prob, True), axis=1)

                with tf.variable_scope('Actor', reuse=tf.AUTO_REUSE):
                    logit       = self._build_a(lstm_out, 'pi', self.eval_bn_flag, self.keep_prob, True)
                    log_logit   = self._build_a(lstm_out, 'old_pi', self.targ_bn_flag, self.keep_prob, False)
                single_gpu_act = tf.reshape(single_gpu_act, (single_gpu_batch_size*traj_length,)) 
                single_gpu_adv = tf.reshape(single_gpu_adv, (single_gpu_batch_size*traj_length,)) 
                single_gpu_ret = tf.reshape(single_gpu_ret, (single_gpu_batch_size*traj_length,)) 

                log_pi     = get_logp_action_mask(single_gpu_act, logits = logit, action_masks = single_gpu_obs2)
                log_old_pi = get_logp_action_mask(single_gpu_act, logits = log_logit, action_masks = single_gpu_obs2)
                log_old_pi = tf.stop_gradient(log_old_pi)

                ratio = tf.exp(log_pi - log_old_pi)
                ratio = tf.clip_by_value(ratio, 0, 3)
                self.check_list.append(logit)
                self.rnn_value_check_list.append(lstm_out)
                self.rnn_value_check_list.append(check_values)
                kl.append(log_old_pi-log_pi)
                min_adv = tf.where(single_gpu_adv>0, (1+clip_ratio)*single_gpu_adv, (1-clip_ratio)*single_gpu_adv)

                entropy = cat_entropy_action_mask(logit, action_masks = single_gpu_obs2)

                pi_loss = -tf.reduce_mean(tf.minimum(ratio * single_gpu_adv, min_adv))-tf.reduce_mean(entropy * ent_rate)
                v_loss  = tf.reduce_mean((single_gpu_ret - critic)**2*0.5)
                loss = pi_loss + v_loss
                losses.append(loss)

                self.train_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Common')+\
                                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Actor/pi')+\
                                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'critic')
                                    
                self.pi_params       = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/pi')
                self.old_pi_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/old_pi')
                with tf.device('/gpu:{}'.format(gpu_list[i])):
                    grads.append(self.optimizer.compute_gradients(loss, self.train_params))

            grad = self.average_gradients(grads)
            grad = self.clip_grad(grad)
            self.train_op = self.optimizer.apply_gradients(grad)
            if use_bn:
                bn_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'Common') +\
                                tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'Actor/pi') +\
                                tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'critic')
                with tf.control_dependencies([tf.group(bn_update_ops, self.train_op)]):
                    self.rnn_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Common/lstm')# , 
                    l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in self.rnn_params if 'bias' not in v.name])
                    sgd = tf.train.GradientDescentOptimizer(learning_rate=1.0)
                    self.weight_decay_op = sgd.minimize(l2_loss)

            else:
                with tf.control_dependencies(self.train_op):
                    self.rnn_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Common/lstm')# , 
                    l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in self.rnn_params if 'bias' not in v.name])
                    sgd = tf.train.GradientDescentOptimizer(learning_rate=1.0)
                    self.weight_decay_op = sgd.minimize(l2_loss)


            self.loss = self.average_loss(losses)
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

        config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess  = tf.Session(config = config)
        
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
        self.sess.run(self.lr_op)

    def check_lr(self):
        lr = self.sess.run(self.optimizer ._lr)
        print('Learning rate',  lr)
        return lr

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

    def learn(self, samples):
        image_ph = samples['image_ph']
        obs1_ph = samples['obs1_ph']
        obs2_ph = samples['obs2_ph']
        act_ph = samples['act_ph']
        adv_ph = samples['adv_ph']
        # actions_ph = samples['actions_ph']
        ret_ph = samples['ret_ph']
        new_traj_start   = samples['new_traj_start']
        batchsize = len(act_ph)
        lstm_init_state = np.zeros((batchsize, lstm_dim*2))
        _, loss = self.sess.run([self.weight_decay_op, self.loss],\
        {self.image:image_ph, self.act:act_ph, self.obs1:obs1_ph, self.obs2:obs2_ph, self.adv_ph:adv_ph, self.ret_ph:ret_ph, self.lstm_init_state:lstm_init_state, \
                    self.new_traj_start:new_traj_start, self.eval_bn_flag:True, self.targ_bn_flag:True, self.keep_prob:0.6})

        return loss

    def rnn_value_check(self, samples):
        image_ph = samples['image_ph']
        obs1_ph = samples['obs1_ph']
        obs2_ph = samples['obs2_ph']
        act_ph = samples['act_ph']
        adv_ph = samples['adv_ph']
        ret_ph = samples['ret_ph']
        new_traj_start   = samples['new_traj_start']
        batchsize = len(act_ph)
        lstm_init_state = np.zeros((batchsize, lstm_dim*2))
        rnn_value_check_list= self.sess.run(self.rnn_value_check_list,\
        {self.image:image_ph, self.act:act_ph, self.obs1:obs1_ph, self.obs2:obs2_ph, self.adv_ph:adv_ph, self.ret_ph:ret_ph, self.lstm_init_state:lstm_init_state, \
                    self.new_traj_start:new_traj_start, self.eval_bn_flag:False, self.targ_bn_flag:True, self.keep_prob:1})
        rnn_value = rnn_value_check_list[0]
        lstm_first_hidden = rnn_value_check_list[1]
        total_value = 0
        for i in range(len(rnn_value)):
            for j in range(len(rnn_value[0])):
                total_value += abs(rnn_value[i][j])
        print('rnn total value check =========================================', total_value)
        print('lstm_first_hidden', lstm_first_hidden)

    def check(self,samples):
        image_ph = samples['image_ph']
        obs1_ph = samples['obs1_ph']
        obs2_ph = samples['obs2_ph']
        act_ph = samples['act_ph']
        adv_ph = samples['adv_ph']
        new_traj_start   = samples['new_traj_start']
        batchsize = len(act_ph)
        lstm_init_state = np.zeros((batchsize, lstm_dim*2))
        check_list = self.sess.run(self.check_list,\
        {self.image:image_ph, self.act:act_ph, self.obs1:obs1_ph, self.obs2:obs2_ph, self.adv_ph:adv_ph, self.lstm_init_state:lstm_init_state, \
                    self.new_traj_start:new_traj_start, self.eval_bn_flag:False, self.targ_bn_flag:True, self.keep_prob:1})
        # print('checkcheckchekc-------------------------')
        # print(check_list)

        lstm_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Common/lstm')
        for p in lstm_params:
            print(p)
            print(self.sess.run(p))

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
        pass
        # for p in self.train_pi_params:
        #     print(p)
        #     print(self.sess.run(p))
        # for p in self.c_params:
        #     print(p)
        #     print(self.sess.run(p))
    
    def update_old_pi(self):#更新旧网络参数，使之与当前的actor完全相同
        self.sess.run(self.set_equal)

    def get_kl(self, image, obs1, obs2, a):
        return self.sess.run(self.kl,{self.image:image, self.obs1:obs1, self.obs2:obs2,self.a:a,\
                                      self.eval_bn_flag:True, self.targ_bn_flag:True})
    #返回一个list
    def get_action(self, image, obs1, obs2):
        return self.sess.run(self.sample_action,{self.image:image, self.obs1:obs1, self.obs2:obs2, self.eval_bn_flag:False})
        
    def get_value(self,image, obs1, obs2, actions):
        return self.sess.run(self.critic,{self.image:image, self.obs1:obs1, self.obs2:obs2, self.actions:actions})
    
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
            
    def restore_model(self, model_path):
            self.saver.restore(self.sess, model_path)

    def _build_cnn(self, image, scope, bn, trainable):
        with tf.variable_scope(scope):
            x = conv2d(image, 32, "l1", [5, 5], [3, 3], pad="SAME")
            if use_bn:
                x  = tf.layers.batch_normalization(x, training=bn)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = conv2d(x, 64, "l2", [4, 4], [2, 2], pad="SAME")
            if use_bn:
                x  = tf.layers.batch_normalization(x, training=bn)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = conv2d(x, 64, "l3", [3, 3], [1, 1], pad="SAME")
            if use_bn:
                x  = tf.layers.batch_normalization(x, training=bn)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.Flatten()(x) 
            return x

    def _build_hidden_state(self, cnn_out, obs1, obs2, scope, bn, keep_prob, trainable):
        init_w = tf.random_normal_initializer(0., 0.01)
        init_b = tf.constant_initializer(0.01)
        with tf.variable_scope(scope):

            x    = tf.concat([obs1, obs2], axis = -1)
            n_s  = hidden_dim
            w_s  = tf.get_variable('w_s', [x.shape[-1], n_s], initializer=init_w, trainable=trainable)
            b_s  = tf.get_variable('b_s', [1, n_s], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w_s) + b_s
            if use_bn:
                x    = tf.layers.batch_normalization(x, training=bn)
            x    = tf.nn.leaky_relu(x, alpha=0.2)
            # if use_dropout:
            #     x    = tf.nn.dropout(x, keep_prob = keep_prob)
            x    = tf.concat([x, cnn_out], axis = -1)

            return x

    def _build_a(self, rnn_out, scope, bn, keep_prob, trainable):
        init_w = tf.random_normal_initializer(0., 0.01)
        init_b = tf.constant_initializer(0.01)
        with tf.variable_scope(scope):

            x     = rnn_out
            
            n_l2 = 256
            w2_s = tf.get_variable('w2_s', [x.shape[-1], n_l2], initializer=init_w, trainable=trainable)
            b2   = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w2_s) + b2
            if use_bn:
                x    = tf.layers.batch_normalization(x, training=bn)
            x    = tf.nn.leaky_relu(x, alpha=0.2)
            # if use_dropout:
            #     x    = tf.nn.dropout(x, keep_prob = keep_prob)

            n_l3 = self.a_dim
            w3_s = tf.get_variable('w3_s', [x.shape[-1], n_l3], initializer=init_w, trainable=trainable)
            b3   = tf.get_variable('b3', [1, n_l3], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w3_s) + b3

            return x
        
    def _build_c(self, rnn_out, actions, scope, bn, keep_prob, trainable):
        init_w = tf.random_normal_initializer(0., 0.01)
        init_b = tf.constant_initializer(0.01)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

            
            # n_l1   = 512
            # n_acts = 64
            # w1_s   = tf.get_variable('w1_s', [x.shape[-1], n_l1], initializer=init_w, trainable=trainable)
            # w_acts = tf.get_variable('w_acts', [actions.shape[-1], n_acts], initializer=init_w, trainable=trainable)
            # b1     = tf.get_variable('b1', [1, n_l1 + n_acts], initializer=init_b, trainable=trainable)
            # x      = tf.concat([tf.matmul(x, w1_s), tf.matmul(actions, w_acts)], axis = -1) + b1
            # x      = tf.nn.leaky_relu(x, alpha=0.2)
            x     = rnn_out
            n_l2 = 256
            w2_s = tf.get_variable('w2_s', [x.shape[-1], n_l2], initializer=init_w, trainable=trainable)
            b2   = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w2_s) + b2
            if use_bn:
                x    = tf.layers.batch_normalization(x, training=bn)
            x    = tf.nn.leaky_relu(x, alpha=0.2)
            # if use_dropout:
            #     x    = tf.nn.dropout(x, keep_prob = keep_prob)

            n_l3 = 1
            w3_s = tf.get_variable('w3_s', [x.shape[-1], n_l3], initializer=init_w, trainable=trainable)
            b3   = tf.get_variable('b3', [1, n_l3], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w3_s) + b3

            return x
        


if __name__ == '__main__':
    single_gpu_batch_size = 2
    tf.set_random_seed(-1)
    np.random.seed(0)
    agent = build_agent_gpu(image_shape = (72, 48, 3), a_dim = 5, s1_dim = 15, s2_dim = 5, single_gpu_batch_size = single_gpu_batch_size, gpu_list = [3],  mouse_num = 2)
    
    samples = dict()
    
    samples['image_ph'] = np.random.rand(single_gpu_batch_size, traj_length, 72, 48, 3)
    samples['obs1_ph'] = np.random.rand(single_gpu_batch_size, traj_length, 15)
    samples['obs2_ph'] = np.random.rand(single_gpu_batch_size, traj_length, 5)
    # samples['image_ph'] = np.ones((single_gpu_batch_size, traj_length, 72, 48, 3))
    # samples['obs1_ph'] = np.ones((single_gpu_batch_size, traj_length, 15))
    # samples['obs2_ph'] = np.ones((single_gpu_batch_size, traj_length, 5))
    samples['act_ph'] = np.ones((single_gpu_batch_size, traj_length))
    samples['ret_ph'] = np.ones((single_gpu_batch_size, traj_length))
    samples['act_ph'][1] = samples['act_ph'][1]*2
    samples['adv_ph'] = np.ones((single_gpu_batch_size, traj_length))*5
    samples['adv_ph'][1] = samples['adv_ph'][1]*(-2)
    samples['new_traj_start'] = np.zeros((single_gpu_batch_size, traj_length))
    agent.check(samples)
    for i in range(20):
        agent.learn(samples)
        agent.check(samples)
            # agent.rnn_value_check(samples)
        # agent.update_old_pi()