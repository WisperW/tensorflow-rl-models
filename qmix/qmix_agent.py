# encoding: utf-8
import tensorflow as tf
import numpy as np
import os

from sampler_files.tf_utils import conv2d
from sampler_files.tf_build_utils import build_cpu_mix, build_q_mix_v0
from sampler_files.config import restart_config, experiment_config
from sampler_files.BuildLstmCell import dynamic_lstm, LSTMCell
lstm_dim = experiment_config.lstm_dim
traj_length = experiment_config.traj_length
build_lstm_cell = LSTMCell
TAU = 0.96
class build_agent_gpu():
    def __init__(self, image_shape = (72, 48, 3), a_dim = 10, s1_dim = 20, s2_dim = 10, single_gpu_batch_size = 1, gpu_list = [0],\
                 num_agents = 2, using_image = True, sess = None):

        self._use_image = using_image
        self.a_dim      = a_dim
        self.num_agents = num_agents
        visible_gpu = ','.join([str(x) for x in gpu_list])
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpu
        with tf.variable_scope('Global', reuse=tf.AUTO_REUSE):
            with tf.device('/cpu:0'):
                self.image      = tf.placeholder(tf.float32, [None, traj_length, num_agents, *image_shape], name = 'image')
                self.next_image = tf.placeholder(tf.float32, [None, traj_length, num_agents, *image_shape], name = 'next_image')
                self.obs1       = tf.placeholder(tf.float32, [None, traj_length, num_agents, s1_dim], name = 'obs1')
                self.next_obs1  = tf.placeholder(tf.float32, [None, traj_length, num_agents, s1_dim], name = 'next_obs1')
                self.obs2       = tf.placeholder(tf.float32, [None, traj_length, num_agents, s2_dim], name = 'obs2')
                self.next_obs2  = tf.placeholder(tf.float32, [None, traj_length, num_agents, s2_dim], name = 'next_obs2')
                self.rews       = tf.placeholder(tf.float32, [None, traj_length], name = 'rews')
                self.acts       = tf.placeholder(tf.int32,   [None, traj_length, num_agents], name = 'acts')
                self.done       = tf.placeholder(tf.float32, [None, traj_length], name = 'done')
                self.new_traj_start  = tf.placeholder(tf.float32,[None, traj_length], name = 'new_traj_start')
                self.lstm_init_state = tf.placeholder(tf.float32,[None, lstm_dim*2], name = 'lstm_init_state')
                self.child_agent_lstm_init_state = tf.placeholder(tf.float32,[None, 2, lstm_dim*2], name = 'child_agent_lstm_init_state')

                tf.add_to_collection('ph', self.image)
                tf.add_to_collection('ph', self.obs1)
                tf.add_to_collection('ph', self.obs2)
                tf.add_to_collection('ph', self.acts)
                self.ep_reward = tf.placeholder(tf.float32)
                self.eval_bn_flag = tf.placeholder(tf.bool)
                self.targ_bn_flag = tf.placeholder(tf.bool)
                self.keep_prob    = tf.placeholder(tf.float32)
        
                grads = []
                _q_losses = []

                self.lr = tf.Variable(5e-5)
                decay = tf.constant(0.95)
                self.lr_op = tf.assign(self.lr, tf.multiply(self.lr, decay))
                self.explore_ratio = tf.Variable(restart_config.explore_ratio, name = 'explore_ratio')
                explore_ratio_decay = tf.constant(0.99)
                self.update_explore_ratio_op = tf.assign(self.explore_ratio, tf.multiply(self.explore_ratio, explore_ratio_decay))
                self.explore_ratio_reset_op  = tf.assign(self.explore_ratio, restart_config.explore_ratio)
                self.optimizer = tf.train.AdamOptimizer(self.lr)
                # self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
                self.check_list = list()
                self.p_list = list()

                for i in range(len(gpu_list)):
                    single_gpu_image   = self.image[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                    single_gpu_obs1   = self.obs1[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                    single_gpu_obs2   = self.obs2[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                    single_gpu_next_image  = self.next_image[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                    single_gpu_next_obs1   = self.next_obs1[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                    single_gpu_next_obs2   = self.next_obs2[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]            
                    single_gpu_acts  = self.acts[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                    single_gpu_rews = self.rews[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                    single_gpu_done = self.done[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                    single_gpu_new_traj_start = self.new_traj_start[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                    single_gpu_lstm_init_state = self.lstm_init_state[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                    single_gpu_child_agent_lstm_init_state = self.child_agent_lstm_init_state[i * single_gpu_batch_size:(i + 1) * single_gpu_batch_size]
                    
                    single_gpu_obs1 = tf.clip_by_value(single_gpu_obs1, -2, 2)
                    
                    single_gpu_image = tf.reshape(single_gpu_image, (single_gpu_batch_size*traj_length, num_agents, *image_shape))
                    single_gpu_obs1 = tf.reshape(single_gpu_obs1, (single_gpu_batch_size*traj_length, num_agents, s1_dim))
                    single_gpu_obs2 = tf.reshape(single_gpu_obs2, (single_gpu_batch_size*traj_length, num_agents, s2_dim))
                    single_gpu_next_image = tf.reshape(single_gpu_next_image, (single_gpu_batch_size*traj_length, num_agents, *image_shape))
                    single_gpu_next_obs1 = tf.reshape(single_gpu_next_obs1, (single_gpu_batch_size*traj_length, num_agents, s1_dim))
                    single_gpu_next_obs2 = tf.reshape(single_gpu_next_obs2, (single_gpu_batch_size*traj_length, num_agents, s2_dim))
                    single_gpu_acts = tf.reshape(single_gpu_acts, (single_gpu_batch_size*traj_length, num_agents)) 
                    single_gpu_rews = tf.reshape(single_gpu_rews, (single_gpu_batch_size*traj_length,)) 
                    single_gpu_done = tf.reshape(single_gpu_done, (single_gpu_batch_size*traj_length,)) 

                    with tf.variable_scope('eval', reuse=tf.AUTO_REUSE):
                        lstm_cell = build_lstm_cell(lstm_dim, 'lstm', self.keep_prob, True)
                        cnn_out_agent_1  = self._build_cnn(single_gpu_image[:, 0] ,'conv', self.eval_bn_flag, True)  
                        cnn_out_agent_2  = self._build_cnn(single_gpu_image[:, 1] ,'conv', self.eval_bn_flag, True) 
                        hidden_state_agent_1 = self._build_hidden_state(
                            cnn_out_agent_1, 
                            single_gpu_obs1[:, 0], 
                            single_gpu_obs2[:, 0], 
                            'hidden', 
                            self.eval_bn_flag, 
                            True
                        )
                        hidden_state_agent_2 = self._build_hidden_state(
                            cnn_out_agent_2, 
                            single_gpu_obs1[:, 1], 
                            single_gpu_obs2[:, 1], 
                            'hidden', 
                            self.eval_bn_flag, 
                            True
                        )
                        hidden_state_agent_1 = tf.reshape(hidden_state_agent_1, (single_gpu_batch_size, traj_length, hidden_state_agent_1.shape[-1]))
                        hidden_state_agent_2 = tf.reshape(hidden_state_agent_2, (single_gpu_batch_size, traj_length, hidden_state_agent_2.shape[-1]))
                        lstm_out_agent_1, lstm_hidden_state_agent_1, _ = dynamic_lstm(
                            lstm_cell, 
                            hidden_state_agent_1, 
                            single_gpu_child_agent_lstm_init_state[:, 0], 
                            single_gpu_new_traj_start
                        )
                        lstm_out_agent_2, lstm_hidden_state_agent_2, _ = dynamic_lstm(
                            lstm_cell, 
                            hidden_state_agent_2, 
                            single_gpu_child_agent_lstm_init_state[:, 1], 
                            single_gpu_new_traj_start
                        )
                        lstm_out_agent_1 = tf.reshape(lstm_out_agent_1, (single_gpu_batch_size*traj_length, lstm_dim))
                        lstm_out_agent_2 = tf.reshape(lstm_out_agent_2, (single_gpu_batch_size*traj_length, lstm_dim))
                        eval_q1 = self._build_dqn(lstm_out_agent_1, 'dqn', self.eval_bn_flag, True)
                        eval_q2 = self._build_dqn(lstm_out_agent_2, 'dqn', self.eval_bn_flag, True)

                        eval_next_cnn_out_agent_1  = self._build_cnn(single_gpu_next_image[:, 0] ,'conv', self.eval_bn_flag, True)  
                        eval_next_cnn_out_agent_2  = self._build_cnn(single_gpu_next_image[:, 1] ,'conv', self.eval_bn_flag, True) 
                        eval_next_hidden_state_agent_1 = self._build_hidden_state(
                            eval_next_cnn_out_agent_1, 
                            single_gpu_next_obs1[:, 0], 
                            single_gpu_next_obs2[:, 0], 
                            'hidden', 
                            self.eval_bn_flag, 
                            True
                        )
                        eval_next_hidden_state_agent_2 = self._build_hidden_state(
                            eval_next_cnn_out_agent_2, 
                            single_gpu_next_obs1[:, 1], 
                            single_gpu_next_obs2[:, 1], 
                            'hidden', 
                            self.eval_bn_flag, 
                            True
                        )
                        eval_next_hidden_state_agent_1 = tf.reshape(eval_next_hidden_state_agent_1, (single_gpu_batch_size, traj_length, eval_next_hidden_state_agent_1.shape[-1]))
                        eval_next_hidden_state_agent_2 = tf.reshape(eval_next_hidden_state_agent_2, (single_gpu_batch_size, traj_length, eval_next_hidden_state_agent_2.shape[-1]))
                        eval_next_lstm_out_agent_1, eval_next_lstm_hidden_state_agent_1, _ = dynamic_lstm(
                            lstm_cell, 
                            eval_next_hidden_state_agent_1, 
                            single_gpu_child_agent_lstm_init_state[:, 0], 
                            single_gpu_new_traj_start
                        )
                        eval_next_lstm_out_agent_2, eval_next_lstm_hidden_state_agent_2, _ = dynamic_lstm(
                            lstm_cell, 
                            eval_next_hidden_state_agent_2, 
                            single_gpu_child_agent_lstm_init_state[:, 1], 
                            single_gpu_new_traj_start
                        )
                        eval_next_lstm_out_agent_1 = tf.reshape(eval_next_lstm_out_agent_1, (single_gpu_batch_size*traj_length, lstm_dim))
                        eval_next_lstm_out_agent_2 = tf.reshape(eval_next_lstm_out_agent_2, (single_gpu_batch_size*traj_length, lstm_dim))
                        
                        next_eval_q1 = self._build_dqn(eval_next_lstm_out_agent_1, 'dqn', self.eval_bn_flag, True)
                        next_eval_q2 = self._build_dqn(eval_next_lstm_out_agent_2, 'dqn', self.eval_bn_flag, True)
                        next_action1 = tf.argmax(next_eval_q1, axis = 1)
                        next_action2 = tf.argmax(next_eval_q2, axis = 1)

                    with tf.variable_scope('target', reuse=tf.AUTO_REUSE):
                        target_lstm_cell = build_lstm_cell(lstm_dim, 'lstm', self.keep_prob, False)

                        next_cnn_out_agent_1  = self._build_cnn(single_gpu_next_image[:, 0] ,'conv', self.targ_bn_flag, False)  
                        next_cnn_out_agent_2  = self._build_cnn(single_gpu_next_image[:, 1] ,'conv', self.targ_bn_flag, False) 
                        next_hidden_state_agent_1 = self._build_hidden_state(
                            next_cnn_out_agent_1, 
                            single_gpu_next_obs1[:, 0], 
                            single_gpu_next_obs2[:, 0], 
                            'hidden', 
                            self.targ_bn_flag, 
                            False
                        )
                        next_hidden_state_agent_2 = self._build_hidden_state(
                            next_cnn_out_agent_2, 
                            single_gpu_next_obs1[:, 1], 
                            single_gpu_next_obs2[:, 1], 
                            'hidden', 
                            self.targ_bn_flag, 
                            False
                        )

                        next_hidden_state_agent_1 = tf.reshape(next_hidden_state_agent_1, (single_gpu_batch_size, traj_length, next_hidden_state_agent_1.shape[-1]))
                        next_hidden_state_agent_2 = tf.reshape(next_hidden_state_agent_2, (single_gpu_batch_size, traj_length, next_hidden_state_agent_2.shape[-1]))
                        lstm_next_out_agent_1, lstm_next_hidden_state_agent_1, _ = dynamic_lstm(
                            lstm_cell, 
                            next_hidden_state_agent_1, 
                            single_gpu_child_agent_lstm_init_state[:, 0], 
                            single_gpu_new_traj_start
                        )
                        lstm_next_out_agent_2, lstm_next_hidden_state_agent_2, _ = dynamic_lstm(
                            lstm_cell, 
                            next_hidden_state_agent_2, 
                            single_gpu_child_agent_lstm_init_state[:, 1], 
                            single_gpu_new_traj_start
                        )
                        lstm_next_out_agent_1 = tf.reshape(lstm_next_out_agent_1, (single_gpu_batch_size*traj_length, lstm_dim))
                        lstm_next_out_agent_2 = tf.reshape(lstm_next_out_agent_2, (single_gpu_batch_size*traj_length, lstm_dim))

                        target_q1 = self._build_dqn(lstm_next_out_agent_1, 'dqn', self.targ_bn_flag, False)
                        target_q2 = self._build_dqn(lstm_next_out_agent_2, 'dqn', self.targ_bn_flag, False)
                        # target_q1 = target_q1 - tf.math.abs(tf.random_normal(tf.shape(target_q1), stddev = 0.2))
                        # target_q2 = target_q2 - tf.math.abs(tf.random_normal(tf.shape(target_q1), stddev = 0.2))

                        sample_target_q1 = tf.reduce_sum(tf.one_hot(next_action1, depth=a_dim) *target_q1, axis=1)
                        sample_target_q2 = tf.reduce_sum(tf.one_hot(next_action2, depth=a_dim) *target_q2, axis=1)
                        reshaped_sample_target_q1 = tf.reshape(sample_target_q1, (-1, 1))
                        reshaped_sample_target_q2 = tf.reshape(sample_target_q2, (-1, 1))


                        # target_q1 = tf.reshape(tf.reduce_max(target_q1, -1), (-1, 1))
                        # target_q2 = tf.reshape(tf.reduce_max(target_q2, -1), (-1, 1))


                    with tf.variable_scope('mixer', reuse=tf.AUTO_REUSE):
                        sampled_q1 = tf.reshape(tf.reduce_sum(tf.one_hot(single_gpu_acts[:, 0], depth=a_dim) *eval_q1,axis=1), (-1, 1))
                        sampled_q2 = tf.reshape(tf.reduce_sum(tf.one_hot(single_gpu_acts[:, 1], depth=a_dim) *eval_q2,axis=1), (-1, 1))
                        
                        eval_q_input  = tf.concat([sampled_q1 , sampled_q2], axis=1)
                        eval_q_input  = tf.reshape(eval_q_input,(-1, 1, num_agents))

                        target_q_input  = tf.concat([reshaped_sample_target_q1 , reshaped_sample_target_q2], axis=1)
                        target_q_input  = tf.reshape(target_q_input,(-1, 1, num_agents))

                        with tf.variable_scope('eval', reuse=tf.AUTO_REUSE):
                            mix_cnn_out1 = self._build_cnn(single_gpu_image[:, 0] ,'conv', self.eval_bn_flag, True)
                            mix_cnn_out2 = self._build_cnn(single_gpu_image[:, 1] ,'conv', self.eval_bn_flag, True)
                            mix_concat_cnn_out = tf.concat([mix_cnn_out1, mix_cnn_out2], axis = -1)
                            mix_hidden_state = self._build_mix_hidden_state(mix_concat_cnn_out, single_gpu_obs1[:, 0], 'mix_hidden', self.eval_bn_flag, True)
                            mix_hidden_state = tf.reshape(mix_hidden_state, (single_gpu_batch_size, traj_length, mix_hidden_state.shape[-1]))
                            eval_lstm_cell = build_lstm_cell(lstm_dim, 'lstm', self.keep_prob, True)
                            lstm_out, lstm_hidden_state, check_values = dynamic_lstm(eval_lstm_cell, mix_hidden_state, single_gpu_lstm_init_state, single_gpu_new_traj_start)
                            lstm_out = tf.reshape(lstm_out, (single_gpu_batch_size*traj_length, lstm_dim))
                            eval_q_total   = build_q_mix_v0(lstm_out, eval_q_input ,'eval',True)

                        with tf.variable_scope('target', reuse=tf.AUTO_REUSE):
                            mix_next_cnn_out1 = self._build_cnn(single_gpu_next_image[:, 0] ,'conv', self.eval_bn_flag, False)
                            mix_next_cnn_out2 = self._build_cnn(single_gpu_next_image[:, 1] ,'conv', self.eval_bn_flag, False)
                            mix_concat_next_cnn_out = tf.concat([mix_next_cnn_out1, mix_next_cnn_out2], axis = -1)
                            mix_next_hidden_state = self._build_mix_hidden_state(mix_concat_next_cnn_out, single_gpu_next_obs1[:, 0], 'mix_hidden', self.eval_bn_flag, False)
                            mix_next_hidden_state = tf.reshape(mix_next_hidden_state, (single_gpu_batch_size, traj_length, mix_hidden_state.shape[-1]))
                            targ_lstm_cell = build_lstm_cell(lstm_dim, 'lstm', self.keep_prob, False)
                            next_lstm_out, next_lstm_hidden_state, next_check_values = dynamic_lstm(targ_lstm_cell, mix_next_hidden_state, single_gpu_lstm_init_state, single_gpu_new_traj_start)
                            next_lstm_out = tf.reshape(next_lstm_out, (single_gpu_batch_size*traj_length, lstm_dim))
                            next_lstm_out = tf.stop_gradient(next_lstm_out)
                            target_q_total = build_q_mix_v0(next_lstm_out, target_q_input,'target',False)

                        self.check_list.append(eval_q_total)
                        self.check_list.append(target_q_total)
                        self.check_list.append(eval_q1)
                        self.check_list.append(eval_q2)
                        q_target = single_gpu_rews + 0.99 * (1- single_gpu_done) * target_q_total
                        self.p_list.append(q_target)
                        q_target = tf.stop_gradient(q_target)
                        self.train_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Global/eval')+\
                                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Global/mixer/eval')
                        with tf.device('/gpu:{}'.format(gpu_list[i])):
                            q_loss     = tf.reduce_mean((q_target- eval_q_total)**2*0.5)
                            _q_losses.append(q_loss)
                            grads.append(self.optimizer.compute_gradients(q_loss, self.train_params))
                # network_functions = {'build_cnn':self._build_cnn, 'build_hidden_state':self._build_hidden_state, 
                #                      'eval_bn':self.eval_bn_flag, 'targ_bn':self.targ_bn_flag,
                #                      'build_q_mix':build_q_mix_v0, 'build_dqn':self._build_dqn}
                # self.td, cpu_eval_q_total, cpu_target_q_total, cpu_eval_q1, cpu_eval_q2, cpu_hidden_state_agent_2\
                # = build_cpu_mix(network_functions, self.image, self.obs1, self.obs2, self.next_image, self.next_obs1, self.next_obs2, \
                #     self.acts, self.rews, self.done, a_dim, num_agents, mix_version = 0)                
            grad = self.average_gradients(grads)
            grad = self.clip_grad(grad)
            update_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'Global/eval')+\
                            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Global/mixer/eval')
            with tf.control_dependencies(update_ops):
                self.train = self.optimizer.apply_gradients(grad)
            self.q_loss = self.average_loss(_q_losses)


        self.eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Global/eval')+\
                                            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Global/mixer/eval')

        self.target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Global/target')+\
                                            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Global/mixer/target')

        self.soft_update_op = [tf.assign(t, e*(1-TAU) + TAU*t)
                                     for e,t in zip(self.eval_params ,self.target_params)]
        self.set_target_equal_op =  [tf.assign(t, e)
                                     for e,t in zip(self.eval_params ,self.target_params)]

        
        # self.check_list.append(cpu_eval_q_total)
        # self.check_list.append(cpu_target_q_total)
        # self.check_list.append(cpu_eval_q1)
        # self.check_list.append(cpu_eval_q2)
        # self.check_list.append(next_action1)
        # self.check_list.append(next_action2)
        

        self.g_p = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Global')
        for p in self.g_p:
            print(p)
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
        self.saver = tf.train.Saver(var_list=tf.global_variables())
        self.sess  = sess
        logs_path="./data"
        self.train_writer = tf.summary.FileWriter( logs_path,
                                            self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def soft_update(self):
        self.sess.run(self.soft_update_op)
        
    def update_lr(self):
        self.sess.run([self.lr_op])

    def check_lr(self):
        lr = self.sess.run(self.optimizer ._lr)
        print('Learning rate',  lr)
        return lr
        
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
        max_grad_norm = 20

        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        return grads_and_var

    def agent_learn(self, samples):
        image = samples['image']
        obs1 = samples['obs1']
        obs2 = samples['obs2']
        next_image = samples['next_image']
        next_obs1 = samples['next_obs1']
        next_obs2 = samples['next_obs2']
        acts= samples['acts']
        rews = samples['rews']
        done = samples['done']
        new_traj_start = samples['new_traj_start']
        batchsize = len(done)
        lstm_init_state = np.zeros((batchsize, lstm_dim*2))
        child_agent_lstm_init_state = np.zeros((batchsize, 2, lstm_dim*2))
        _, q_loss = self.sess.run([self.train, self.q_loss],\
        {self.rews:rews,self.image:image, self.obs1:obs1, self.obs2:obs2, self.acts:acts, self.done:done, self.new_traj_start:new_traj_start, \
        self.next_image:next_image, self.next_obs1:next_obs1, self.next_obs2:next_obs2, self.lstm_init_state:lstm_init_state, \
        self.child_agent_lstm_init_state:child_agent_lstm_init_state,\
        self.eval_bn_flag:True, self.targ_bn_flag:True})
        return q_loss
       
    def get_td_error(self, image, obs1, obs2, next_image, next_obs1, next_obs2, acts, rews, done):
        pass
        # td = self.sess.run(self.td, {self.rews:rews,self.image:image, self.obs1:obs1, self.obs2:obs2, self.acts:acts, self.done:done, \
        # self.next_image:next_image, self.next_obs1:next_obs1, self.next_obs2:next_obs2, self.eval_bn_flag:True, self.targ_bn_flag:True})
        # return td

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

    def set_equal(self):
        self.sess.run(self.set_target_equal_op)

    
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
            
    def restore_model(self, model_path):
            self.saver.restore(self.sess, model_path)


    def check(self, image, obs1, obs2, next_image, next_obs1, next_obs2, acts, rews, done):
        check_list = self.sess.run(self.check_list,\
        {self.rews:rews,self.image:image, self.obs1:obs1, self.obs2:obs2, self.acts:acts, self.done:done, \
        self.next_image:next_image, self.next_obs1:next_obs1, self.next_obs2:next_obs2})
        print('eval_q_total', check_list[0])
        print('target_q_total', check_list[1])
        print('eval_q1', check_list[2])
        print('eval_q2', check_list[3])
        eval_q_total = check_list[0]
        target_q_total = check_list[1]

        print('cpu_eval_q_total', check_list[4])
        print('cpu_target_q_total', check_list[5])
        print('cpu_eval_q1', check_list[6])
        print('cpu_eval_q2', check_list[7])

        print('act1', check_list[8])
        print('act2', check_list[9])

        p_list= self.sess.run(self.p_list,\
        {self.rews:rews,self.image:image, self.obs1:obs1, self.obs2:obs2, self.acts:acts, self.done:done, \
        self.next_image:next_image, self.next_obs1:next_obs1, self.next_obs2:next_obs2})
        print('cal target', p_list[0])
        # print('cpu_eval_q_total', c3)
        # print('qeval', c7)
        # print('qtarget', c8)
        # print('qloss in ', c9)
        # print('td', td)

        # print('cpu_target_q_total', c4)
        # self.c1 = cpu_eval_q1
        # self.c2 = cpu_eval_q2
        # self.c3 = cpu_eval_q_total
        # self.c4 = cpu_target_q_total
        return eval_q_total, target_q_total

    def reset_explore_ratio(self):
        self.sess.run(self.explore_ratio_reset_op)

    def update_explore_ratio(self):
        self.sess.run(self.update_explore_ratio_op)

    def get_explore_ratio(self):
        return self.sess.run(self.explore_ratio)

    def check_params(self):
        for i, ppp in enumerate(self.eval_params):
            # if i==0:
                print('ppp', ppp)
                print(self.sess.run(ppp))
        for i, ppp in enumerate(self.target_params):
            # if i==0:
                print('ppp', ppp)
                print(self.sess.run(ppp))


    def _build_cnn(self, image, scope, bn, trainable):
        with tf.variable_scope(scope):
            x = conv2d(image, 16, "l1", [5, 5], [3, 3], pad="VALID")
            x = tf.layers.batch_normalization(x, training=bn)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID")
            x = tf.layers.batch_normalization(x, training=bn)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = conv2d(x, 32, "l3", [3, 3], [1, 1], pad="VALID")
            x = tf.layers.batch_normalization(x, training=bn)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.Flatten()(x) 
            return x

    def _build_mix_hidden_state(self, cnn_out, obs1, scope, bn, trainable):
        init_w = tf.random_normal_initializer(0., 0.01)
        init_b = tf.constant_initializer(0.1)
        with tf.variable_scope(scope):

            x     = obs1
            n_s   = 1024
            w_s   = tf.get_variable('w_s', [x.shape[-1], n_s], initializer=init_w, trainable=trainable)
            b_s   = tf.get_variable('b_s', [1, n_s], initializer=init_b, trainable=trainable)
            x     = tf.matmul(x, w_s) + b_s
            x     = tf.layers.batch_normalization(x, training=bn)
            x     = tf.nn.leaky_relu(x, alpha=0.2)
            x     = tf.concat([x, cnn_out], axis = -1)

            return x

    def _build_hidden_state(self, cnn_out, obs1, obs2, scope, bn, trainable):
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

            return x

    def _build_dqn(self, hidden_state, scope, bn, trainable):
        init_w = tf.random_normal_initializer(0., 0.01)
        init_b = tf.constant_initializer(0.01)
        with tf.variable_scope(scope):

            x    = hidden_state
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

    

if __name__ == '__main__':
    config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess  = tf.Session(config = config)
    agent = build_agent_gpu(sess = sess, a_dim = 10, s1_dim = 20, s2_dim = 10)
    # agent.restore_model('test/model')
    # agent.check_params()
    # agent.set_target_equal()
    # agent.check_params()
    agent.check_params()
    image, obs1, obs2, acts, r, done = np.ones((100, 2, 72, 48, 3)), np.ones((100, 2, 20)),  np.ones((100, 2, 10)), np.ones((100, 2)), np.ones(100), np.ones(100)
    next_image, next_obs1, next_obs2 = np.ones((100, 2, 72, 48, 3)), np.ones((100, 2, 20)),  np.ones((100, 2, 10))
    r[2] = 50
    image, obs1, obs2, acts, r, done = np.ones((1, 2, 72, 48, 3)), np.ones((1, 2, 20)),  np.ones((1, 2, 10)), np.ones((1, 2)), np.ones(1)*0.1, [0]
    next_image, next_obs1, next_obs2 = np.ones((1, 2, 72, 48, 3)), np.ones((1, 2, 20)),  np.ones((1, 2, 10))


    new_image, new_obs1, new_obs2, new_acts, new_r, new_done = np.ones((1, 2, 72, 48, 3)), np.ones((1, 2, 20)),  np.ones((1, 2, 10)), [[5, 1]], [5], [1]
    new_next_image, new_next_obs1, new_next_obs2 = np.ones((1, 2, 72, 48, 3)), np.ones((1, 2, 20)),  np.ones((1, 2, 10))
    new_image[:, 1] = new_image[:, 1] *0.5
    new_obs1[:, 1] = new_obs1[:, 1] *0.5
    new_obs2[:, 1] = new_obs2[:, 1] *0.5

    best_acts, best_r = [[1, 2]],  [10]
    # agent.check_params()
    td = agent.get_td_error(image, obs1, obs2, next_image, next_obs1, next_obs2, acts, r, done)
    print('td0', td)
    # print('tdshape1', len(td))
    for i in range(8000):
        agent.agent_learn(image, obs1, obs2, next_image, next_obs1, next_obs2, acts, r, done)
        # good_q_loss = agent.agent_learn(new_image, new_obs1, new_obs2, new_next_image, new_next_obs1, new_next_obs2, new_acts, new_r, new_done)
        # best_q_loss = agent.agent_learn(new_image, new_obs1, new_obs2, new_next_image, new_next_obs1, new_next_obs2, best_acts, best_r, new_done)
        # print('good_q_loss', good_q_loss)
        # print('best_q_loss', best_q_loss)
        print('laji')
        agent.check(image, obs1, obs2, next_image, next_obs1, next_obs2, acts, r, done)
        td = agent.get_td_error(image, obs1, obs2, next_image, next_obs1, next_obs2, acts, r, done)
        print('before', td)
        if i % 40 == 0:
            # eval_q_total, target_q_total = agent.check(image, obs1, obs2, next_image, next_obs1, next_obs2, acts, r, done)
            # if target_q_total[0] > eval_q_total[0]:
            #     print('full update!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            #     agent.set_target_equal()
            # else:
            #     print('soft update=========================================================================================================================')
            agent.soft_update()
        td = agent.get_td_error(image, obs1, obs2, next_image, next_obs1, next_obs2, acts, r, done)
        print('after', td)
        # print('should be good')
        # agent.check(new_image, new_obs1, new_obs2, new_next_image, new_next_obs1, new_next_obs2, new_acts, new_r, new_done)
        # print('best')
        # agent.check(new_image, new_obs1, new_obs2, new_next_image, new_next_obs1, new_next_obs2, [[2, 2]], best_r, new_done)
        # agent.check(new_image, new_obs1, new_obs2, new_next_image, new_next_obs1, new_next_obs2, best_acts, best_r, new_done)
    # agent.check_params()
    # agent.check(image, obs1, obs2, next_image, next_obs1, next_obs2, acts, r, done)
    # td = agent.get_td_error(image, obs1, obs2, next_image, next_obs1, next_obs2, acts, r, done)
    # print('td1', td)
    # # agent.check_params()
    agent.check_params()
    # agent.soft_update()
    # agent.soft_update()
    # # agent.check_params()
    # agent.check(image, obs1, obs2, next_image, next_obs1, next_obs2, acts, r, done)
    # td = agent.get_td_error(image, obs1, obs2, next_image, next_obs1, next_obs2, acts, r, done)
    # print('td2', td)
    agent.save_model('test/model')
