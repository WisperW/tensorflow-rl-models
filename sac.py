# encoding: utf-8
import tensorflow as tf
import numpy as np
TAU = 0.995
lr_c = 5e-4
lr_a = 5e-4
GAMA = 0.99
LOG_STD_MAX = 2
LOG_STD_MIN = -20
alpha = 0.2
EPS = 1e-8
class build_agent():
    def __init__(self, action_dim, state_dim, player):
        self.action_dim = action_dim
        self.state_dim  = state_dim
        self.S           = tf.placeholder(tf.float32, [None, state_dim])
        self.S_          = tf.placeholder(tf.float32, [None, state_dim])
        self.done        = tf.placeholder(tf.float32,(None,))
        self.r           = tf.placeholder(tf.float32,(None,))
        self.acts        = tf.placeholder(tf.float32, [None, action_dim])
        self.sess=tf.Session()
        scope_=str(player)#针对多智能体情况的扩展
        
        with tf.variable_scope(scope_+'Actor'):
            mu, log_std_init  = self._build_a(self.S , 'eval', True)
            
        log_std      = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std_init + 1)
        std          = tf.exp(log_std)
        pi           = mu + tf.random_normal(tf.shape(mu)) * std
        pre_sum      = -0.5 * (((pi-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
        #pre_sum计算该动作在高斯分布中被采样的概率 logp_p pi(a|s)
        logp_pi      = tf.reduce_sum(pre_sum, axis=1)#axis=1的维度会被消掉
        logp_pi     -= tf.reduce_sum(tf.log(self.clip_but_pass_gradient(1-tf.tanh(pi)**2, l=0, u=1) + 1e-6), axis=1)
        self.a       = tf.tanh(mu)#deterministic_action
        self.pi      = tf.tanh(pi)#random_action
            
            
            
        with tf.variable_scope(scope_+'value'):
            self.q1_pi  = self._build_q(self.S, self.pi, 'Q1', True)
            self.q2_pi  = self._build_q(self.S, self.pi, 'Q2', True)
        with tf.variable_scope(scope_+'value', reuse = True):
            self.q1  = self._build_q(self.S, self.acts, 'Q1', True)
            self.q2  = self._build_q(self.S, self.acts, 'Q2', True)
        with tf.variable_scope(scope_+'value'):
            self.v   = self._build_v(self.S , 'eval', True)
        with tf.variable_scope(scope_):
            self.v_  = self._build_v(self.S_, 'targ', False)
            
            
        
        self.ae_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_+'Actor/eval')
        self.q1_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_+'Q/1')
        self.q2_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_+'Q/2')#
        self.ve_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_+'value/eval')#
        self.vt_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_+'/targ')
        self.v_total_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_+'value')

        min_q_pi = tf.minimum(self.q1_pi, self.q2_pi)
        v_target = tf.stop_gradient(min_q_pi - alpha*logp_pi)
        q_target = tf.stop_gradient(self.r + GAMA*self.v_)#*(1-self.done))
        q1_loss  = 0.5 * tf.reduce_mean((q_target - self.q1)**2)
        q2_loss  = 0.5 * tf.reduce_mean((q_target - self.q2)**2)
        v_loss   = 0.5 * tf.reduce_mean((v_target - self.v)**2)
        a_loss   = tf.reduce_mean(alpha*logp_pi - self.q1_pi)

        value_loss = q1_loss + q2_loss + v_loss

        tf.summary.scalar('v_loss',v_loss)
        tf.summary.scalar('a_loss',a_loss)
        self.merged = tf.summary.merge_all()
        logs_path="./data"
        self.train_writer = tf.summary.FileWriter( logs_path,
                                              self.sess.graph)

        a_op = tf.train.AdamOptimizer(lr_a)
        self.grad_a = a_op.compute_gradients(a_loss, var_list=self.ae_params)
        self.a_train = a_op.apply_gradients(self.grad_a)
        
        v_op  = tf.train.AdamOptimizer(lr_c)
        self.grad_v = v_op.compute_gradients(value_loss, var_list=self.v_total_params)
        self.v_train = v_op.apply_gradients(self.grad_v)
            
        self.soft_replace = [tf.assign(t, (1 - TAU) * e + TAU *t)
                                     for e, t in zip(self.ve_params, self.vt_params)]
        

        #希望查看的值列表
        self.c1 =self.q1_pi
        self.c2 =self.q2_pi
        self.c3 =self.v
        self.c4 =self.v_
        self.c5 =pi
        self.c6 =mu
        self.c7=logp_pi
        self.c8=log_std_init
        self.c9  = (1-self.done)
        self.c10 = pre_sum
        
        self.saver = tf.train.Saver(var_list=tf.global_variables())
        
        self.sess.run(tf.global_variables_initializer())
        
    def actor_learn(self,obs1,obs2,acts,rews,done):
        self.sess.run(self.a_train,{self.S:obs1})
        #print('grad_a',self.sess.run(self.c9,{self.S:obs1})[0])
 
    def critic_learn(self, obs1, obs2, acts, rews, done ):
        self.sess.run(self.v_train,{self.S:obs1, self.S_:obs2, self.acts:acts, self.done:done,
                                  self.r:rews})  

    def get_action(self, s, is_train):

        if is_train:
            return self.sess.run(self.pi, {self.S:s})[0]
        else:
            return self.sess.run(self.a,  {self.S:s})[0]
        
    def clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = tf.cast(x > u, tf.float32)
        clip_low = tf.cast(x < l, tf.float32)
        return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)
        

    def check_value(self,s):

        
#        print('q1_pi',self.sess.run(self.c1,{self.S:s})[0])
#        print('q2_pi',self.sess.run(self.c2,{self.S:s})[0])
#        print('v',self.sess.run(self.c3,{self.S:s})[0])
#        print('v_',self.sess.run(self.c4,{self.S_:s})[0])
#        print('pi',self.sess.run(self.c5,{self.S:s})[0])
#        print('mu',self.sess.run(self.c6,{self.S:s})[0])
#        print('logp_pi',self.sess.run(self.c7,{self.S:s})[0])
        print('log_std',self.sess.run(self.c8,{self.S:s})[0])
#        print('logp_pi',self.sess.run(self.c10,{self.S:s})[0])

   
    def check_avalue(self,s,a):
        return self.sess.run(self.q1,{self.S:s,self.acts:a})[0]
    
    def save_tensor(self, obs1, obs2, acts, rews, done, T):
        summary=self.sess.run(self.merged,{self.S:obs1, self.S_:obs2, self.acts:acts, self.done:done,
                                  self.r:rews})
        self.train_writer.add_summary(summary, T)
    
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
            
    def restore_model(self, model_path):
            self.saver.restore(self.sess, model_path)
            
    def update_t_n(self):
            self.sess.run(self.soft_replace)

        
    def _build_a(self, s, scope, trainable):
        init_w = tf.random_normal_initializer(0., 0.1)
        init_b = tf.random_normal_initializer(0., 0.1)
        with tf.variable_scope(scope):
            n_l1 = 400
            w1_s = tf.get_variable('w1_s', [self.state_dim, n_l1], initializer=init_w, trainable=trainable)
            b1   = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
            x    = tf.nn.tanh(tf.matmul(s, w1_s) + b1)
            tf.summary.histogram('a/w1_s',w1_s)
            
            n_l2 = 300
            w2_s = tf.get_variable('w2_s', [n_l1, n_l2], initializer=init_w, trainable=trainable)
            b2   = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
            x    = tf.nn.tanh(tf.matmul(x, w2_s) + b2)
            tf.summary.histogram('a/w2_s',w2_s)

            n_l3 = self.action_dim
            w3_s = tf.get_variable('w3_s', [n_l2, n_l3], initializer=init_w, trainable=trainable)
            b3   = tf.get_variable('b3', [1, n_l3], initializer=init_b, trainable=trainable)
            mu   = tf.matmul(x, w3_s) + b3
            tf.summary.histogram('a/w3_s',w3_s)
            
            n_l4 = self.action_dim
            w4_s = tf.get_variable('w4_s', [n_l2, n_l4], initializer=init_w, trainable=trainable)
            b4   = tf.get_variable('b4', [1, n_l4], initializer=init_b, trainable=trainable)
            std  = tf.nn.tanh(tf.matmul(x, w4_s) + b4)
            tf.summary.histogram('a/w4_s',w4_s)
            
            return mu, std
        

    def _build_q(self, s, a, scope, trainable):
        
        init_w = tf.random_normal_initializer(0., 0.1)
        init_b = tf.random_normal_initializer(0., 0.1)
        with tf.variable_scope(scope):
            n_l1 = 400
            x    = tf.concat([s,a], axis = 1)
            w1_s = tf.get_variable('w1_s', [self.state_dim+self.action_dim, n_l1], initializer=init_w, trainable=trainable)
            b1   = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w1_s)+b1
            x    = tf.nn.relu(x)
            tf.summary.histogram('q/w1_s',w1_s)
            
            n_l2 = 300
            w2_s = tf.get_variable('w2_s', [n_l1, n_l2], initializer=init_w, trainable=trainable)
            b2   = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w2_s)+b2
            x    = tf.nn.relu(x)
            tf.summary.histogram('q/w2_s',w2_s)
            
            n_l3 = 1
            w3_s = tf.get_variable('w3_s', [n_l2, n_l3], initializer=init_w, trainable=trainable)
            b3   = tf.get_variable('b3', [1, n_l3], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w3_s)+b3
            tf.summary.histogram('q/w3_s',w3_s)
            x    = tf.squeeze(x, axis=1) #非常重要
            return x
        
    def _build_v(self, s, scope, trainable):
        
        init_w = tf.random_normal_initializer(0., 0.1)
        init_b = tf.random_normal_initializer(0., 0.1)
        with tf.variable_scope(scope):
            n_l1 = 400
            w1_s = tf.get_variable('w1_s', [self.state_dim, n_l1], initializer=init_w, trainable=trainable)
            b1   = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
            x    = tf.matmul(s, w1_s)+b1
            x    = tf.nn.relu(x)
            
            n_l2 = 300
            w2_s = tf.get_variable('w2_s', [n_l1, n_l2], initializer=init_w, trainable=trainable)
            b2   = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w2_s)+b2
            x    = tf.nn.relu(x)
            tf.summary.histogram('v/w2_s',w2_s)
            
            n_l3 = 1
            w3_s = tf.get_variable('w3_s', [n_l2, n_l3], initializer=init_w, trainable=trainable)
            b3   = tf.get_variable('b3', [1, n_l3], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w3_s)+b3
            x    = tf.squeeze(x, axis=1)
            return x