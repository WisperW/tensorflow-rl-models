
import tensorflow as tf
import numpy as np
from ReplayBuffer import ReplayBuffer as buffer_without_td
from ReplayBuffer_td import ReplayBuffer as buffer_with_td
TAU=0.96
lr_c=0.0001
lr_a=0.0001
GAMA=0.99
a_bound = 10
act_noise=0.1
target_noise=0.2
noise_clip=0.5
batch_size = 128
replayBuffer_size = 200000
class td3():
    def __init__(self, action_dim, state_dim, using_td, player):
        self.action_dim = action_dim
        self.state_dim  = state_dim
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.using_td = using_td
        self.S           = tf.placeholder(tf.float32, [None, state_dim])
        self.S_          = tf.placeholder(tf.float32, [None, state_dim])
        self.done        = tf.placeholder(tf.float32,(None,))
        self.r           = tf.placeholder(tf.float32,(None,))
        self.average_reward = tf.placeholder(tf.float32)
        if using_td ==False:
            self.buffer = buffer_without_td(state_dim, action_dim, replayBuffer_size)
        else:
            self.buffer = buffer_with_td(state_dim, action_dim, replayBuffer_size)

        self.sess=tf.Session()
        scope_=str(player)
        with tf.variable_scope(scope_+'Actor'):
            self.actor  = self._build_a(self.S , 'eval', True)
            self.actor_ = self._build_a(self.S_, 'target', False)
        epsilon = tf.random_normal(tf.shape(self.actor_), stddev=target_noise)
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)*a_bound
        a2 = self.actor_ + epsilon
        a2 = tf.clip_by_value(a2, -a_bound, a_bound)
        with tf.variable_scope(scope_+'Critic'):
            self.q1  = self._build_c(self.S ,self.actor ,'q1_eval', True) 
            self.q2  = self._build_c(self.S ,self.actor ,'q2_eval', True) 
            self.q1_ = self._build_c(self.S_,a2,'q1_target', False)
            self.q2_ = self._build_c(self.S_,a2,'q2_target', False)
            
        
        self.ae_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_+'Actor/eval')
        self.at_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_+'Actor/target')
        self.q1e_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_+'Critic/q1_eval')
        self.q1t_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_+'Critic/q1_target')
        self.q2e_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_+'Critic/q2_eval')
        self.q2t_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_+'Critic/q2_target')
        self.soft_replace1 = [tf.assign(t, (1 - TAU) * e + TAU *t)
                                     for e,t in zip(self.ae_params,self.at_params)]
        self.soft_replace2 = [tf.assign(t, (1 - TAU) * e + TAU *t)
                                     for e,t in zip(self.q1e_params,self.q1t_params)]
        self.soft_replace3 = [tf.assign(t, (1 - TAU) * e + TAU *t)
                                     for e,t in zip(self.q2e_params,self.q2t_params)]
        self.q_target = self.r+GAMA*tf.minimum(self.q1_, self.q2_)*(1-self.done)
        self.td1_error = tf.reduce_mean(tf.square(self.q_target-self.q1))
        self.td2_error = tf.reduce_mean(tf.square(self.q_target-self.q2))
        self.value_loss = self.td1_error+self.td2_error
        a_loss = tf.reduce_mean(tf.negative(self.q1))

        a_op         = tf.train.AdamOptimizer(lr_a)
        grad_a       = a_op.compute_gradients(a_loss, var_list=self.actor)
        up_grad      = tf.multiply(grad_a[0][0], (a_bound-self.actor)/a_bound)
        down_grad    = tf.multiply(grad_a[0][0], (a_bound+self.actor)/a_bound)
        invert_grad  = tf.where(grad_a[0][0]<0, up_grad, down_grad)
        #invert_grad  = tf.where(self.actor+a_bound<0, down_grad, temp_invert_grad)
        grad         = a_op.compute_gradients(self.actor, var_list=self.ae_params, grad_loss=invert_grad)# invert_grad)

        self.a_train = a_op.apply_gradients(grad)
        self.c_train = tf.train.AdamOptimizer(lr_c).minimize(self.value_loss, var_list=self.q1e_params+self.q2e_params)
        
        tf.summary.scalar('average_reward',self.average_reward)
        tf.summary.scalar('grad_for_a',tf.reduce_mean(grad[0][0]))
        tf.summary.scalar('value_loss',self.value_loss)
        tf.summary.scalar('td1_error',self.td1_error)
        tf.summary.scalar('td2_error',self.td2_error)
        self.merged = tf.summary.merge_all()
        logs_path="./data"
        self.train_writer = tf.summary.FileWriter( logs_path,
                                              self.sess.graph)

        self.c1  = self.actor
        self.c2  = self.actor_
        self.c3 =self.q1
        self.c4 =self.q2
        self.c5 =self.q1_
        self.c6 =self.q2_
        self.c7 = a2

        self.saver = tf.train.Saver(var_list=tf.global_variables())
        
        self.sess.run(tf.global_variables_initializer())
        
    def actor_learn(self,obs1):
        self.sess.run(self.a_train,{self.S:obs1})
    
    def critic_learn(self, obs1, obs2, acts, rews, done):
        self.sess.run(self.c_train,{self.S:obs1,self.S_:obs2,self.actor:acts*a_bound,self.done:done,
                                  self.r:rews}) 

    def store(self, obs1, acts, rews, obs2, done):
        if self.using_td:
            td = self.get_td(obs1, obs2, acts, rews, done)
            self.buffer.store(obs1, acts, rews, obs2, done, td)
        else:
            self.buffer.store(obs1, acts, rews, obs2, done)
            
    def train(self):
        delay_step = 2
        if self.using_td:
            for i in range(2):
                data, idxs = self.buffer.sample_batch(batch_size) 
                self.critic_learn(data['obs1'], data['obs2'],data['acts'], data['rews'], data['done'])
                if i % delay_step ==0:
                    self.actor_learn(data['obs1'])
                    self.update_t_n()
                td = self.get_td(data['obs1'], data['obs2'],data['acts'], data['rews'], data['done'])
                self.buffer.update_td(idxs, td)
        else:
            for i in range(2):
                data = self.buffer.sample_batch(batch_size) 
                self.critic_learn(data['obs1'], data['obs2'],data['acts'], data['rews'], data['done'])
                if i % delay_step ==0:
                    self.actor_learn(data['obs1'])
                    self.update_t_n()


    def get_td(self, obs1, obs2, acts, rews, done):
        return self.sess.run(self.td1_error,{self.S:obs1,self.S_:obs2,self.actor:acts*a_bound,self.done:done,
                                  self.r:rews})   

    def get_action(self,s):
        return self.sess.run(self.actor,{self.S:s})[0]/a_bound
    
    def save_tensor(self, obs1, obs2, acts, rews ,done, average_reward, episode):
        summary=self.sess.run(self.merged,{self.S:obs1, self.S_:obs2 ,self.actor:acts*a_bound, self.done:done,
                                  self.r:rews, self.average_reward:average_reward})
        self.train_writer.add_summary(summary, episode)    
        
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
            
    def restore_model(self, model_path):
            self.saver.restore(self.sess, model_path)
            
    def update_t_n(self):
            self.sess.run(self.soft_replace1)
            self.sess.run(self.soft_replace2)
            self.sess.run(self.soft_replace3)
    def check_a_value(self, s, a):
        return self.sess.run(self.q1,{self.S:s, self.actor:a})  
            
    def check_c(self,s):
        return [self.sess.run(self.c1,{self.S:s, self.S_:s}),
                self.sess.run(self.c2,{self.S:s, self.S_:s}),
                self.sess.run(self.c3,{self.S:s, self.S_:s}),
                self.sess.run(self.c4,{self.S:s, self.S_:s}),
                self.sess.run(self.c5,{self.S:s, self.S_:s}),
                self.sess.run(self.c6,{self.S:s, self.S_:s}),
                self.sess.run(self.c7,{self.S:s, self.S_:s})
                ]
        
    def _build_a(self, s, scope, trainable):

        init_w = tf.random_normal_initializer(0., 0.1)
        init_b = tf.constant_initializer(0.1)
        with tf.variable_scope(scope):
            n_l1 = 512
            w1_s = tf.get_variable('w1_s', [self.state_dim, n_l1], initializer=init_w, trainable=trainable)
            b1   = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
            x    = tf.nn.tanh(tf.matmul(s, w1_s) + b1)
            
            n_l2 = 512
            w2_s = tf.get_variable('w2_s', [n_l1, n_l2], initializer=init_w, trainable=trainable)
            b2   = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
            x    = tf.nn.tanh(tf.matmul(x, w2_s) + b2)
#            x = tf.layers.batch_normalization(s, training=self.batch_train)
            n_l3 = self.action_dim
            w3_s = tf.get_variable('w3_s', [n_l2, n_l3], initializer=init_w, trainable=trainable)
            b3   = tf.get_variable('b3', [1, n_l3], initializer=init_b, trainable=trainable)
            a    = tf.matmul(x, w3_s) + b3
            #a    = tf.nn.tanh(x)
            return a
        

    def _build_c(self, s, a, scope, trainable):
        
        init_w = tf.random_normal_initializer(0., 0.1)
        init_b = tf.constant_initializer(0.1)
        with tf.variable_scope(scope):
            n_l1 = 512
            w1_s = tf.get_variable('w1_s', [self.state_dim, n_l1], initializer=init_w, trainable=trainable)
            b1   = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
            x    = tf.matmul(s, w1_s)+b1
            x    = tf.nn.relu(x)
            
            n_l2 = 512
            n_la = self.action_dim
            w2_s = tf.get_variable('w2_s', [n_l1, n_l2], initializer=init_w, trainable=trainable)
            w2_a = tf.get_variable('w2_a', [n_la, n_l2], initializer=init_w, trainable=trainable)
            b2   = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w2_s)+tf.matmul(a, w2_a)+b2
            x    = tf.nn.relu(x)
            
            n_l3 = 1
            w3_s = tf.get_variable('w3_s', [n_l2, n_l3], initializer=init_w, trainable=trainable)
            b3   = tf.get_variable('b3', [1, n_l3], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w3_s)+b3
            return tf.squeeze(x, axis=1)

