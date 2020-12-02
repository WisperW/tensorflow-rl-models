# encoding: utf-8
import tensorflow as tf
import numpy as np
import math
import random
import os
TAU = 0.96
vmax = 10
vmin = -1.0
num_atoms = 23
class build_agent():
    def __init__(self, a_dim, s_dim, gpu_list, sess, is_gpu = True):
        if is_gpu:
            visible_gpu = ','.join([str(x) for x in gpu_list])
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpu
        with tf.device('/cpu:0'):
            self.a_dim           = a_dim
            self.obs_ph          = tf.placeholder(tf.float32, [None, s_dim])
            self.next_obs_ph     = tf.placeholder(tf.float32, [None, s_dim])
            self.act_ph          = tf.placeholder(tf.int32, (None,))
            self.rew_ph          = tf.placeholder(tf.float32, (None,))
            self.done_ph         = tf.placeholder(tf.float32, (None,))
            self.eval_bn_flag    = tf.placeholder(tf.bool)
            self.targ_bn_flag    = tf.placeholder(tf.bool)
            self.num_atoms = num_atoms
            self._support = tf.linspace(vmin, vmax, num_atoms)
            self.cumulative_gamma = 0.99
            self.epsilon_eval = 0.001
            self.epsilon_decay_period = 250000
            self.training_steps = 0
            self.min_replay_history = 10000
            self.epsilon_train = 0.01

            self.sess  = sess

            self.lr = tf.Variable(1e-3)
            decay = tf.constant(0.95)
            self.lr_update_op = tf.assign(self.lr, tf.multiply(self.lr, decay))
            self.optimizer = tf.train.AdamOptimizer(self.lr) 
            # tf.train.RMSPropOptimizer(
                                # learning_rate=0.00025,
                                # decay=0.95,
                                # momentum=0.0,
                                # epsilon=0.00001,
                                # centered=True)
            if is_gpu:
                num_device = len(gpu_list)
            else:
                num_device = 1

            single_device_obs_ph          = tf.split(self.obs_ph, num_device)
            single_device_next_obs_ph     = tf.split(self.next_obs_ph, num_device)
            single_device_act_ph          = tf.split(self.act_ph, num_device)
            single_device_rew_ph          = tf.split(self.rew_ph, num_device)
            single_device_done_ph         = tf.split(self.done_ph, num_device)
            grads = []
            losses = []
            self.check_list = list()
            single_gpu_batch_size = tf.shape(single_device_obs_ph[0])[0]
            # print('shape', sess.run(single_gpu_batch_size, {self.obs_ph:np.zeros((2, 10))}))
            for i in range(num_device):
                with tf.variable_scope('main', reuse=tf.AUTO_REUSE):

                    distributed_q_logits = self._build_net(
                        single_device_obs_ph[i], 'eval', self.eval_bn_flag, True)
                    distributed_q_probabilities  = tf.nn.softmax(distributed_q_logits)

                    eval_q_values = tf.reduce_sum(tf.multiply(distributed_q_probabilities, self._support), axis=-1)
                    self._q_argmax = tf.squeeze(tf.argmax(eval_q_values, axis=-1))

                    self.check_list.append(distributed_q_probabilities)
                    eval_distributed_next_q_logits  = self._build_net(
                        single_device_next_obs_ph[i], 'eval', self.eval_bn_flag, True)
                    eval_distributed_next_q_probabilities  = tf.nn.softmax(eval_distributed_next_q_logits)

                    eval_next_q_values = tf.reduce_sum(tf.multiply(eval_distributed_next_q_probabilities, self._support), axis=-1)
                    next_qt_argmax = tf.squeeze(tf.argmax(eval_next_q_values, axis=-1))
                    
                    targ_distributed_next_q_logits = self._build_net(
                        single_device_next_obs_ph[i], 'targ', self.targ_bn_flag, False)
                    targ_distributed_next_q_probabilities    = tf.nn.softmax(targ_distributed_next_q_logits)

                    # size of next_qt_argmax: 1 x batch_size
                    next_qt_argmax = tf.expand_dims(next_qt_argmax, -1)
                    batch_indices = tf.expand_dims(tf.range(tf.cast(single_gpu_batch_size, tf.int64)), -1)
                    # size of next_qt_argmax: batch_size x 2
                    batch_indexed_next_qt_argmax = tf.concat(
                        [batch_indices, next_qt_argmax], axis=1)

                    # size of next_probabilities: batch_size x num_atoms
                    next_probabilities = tf.gather_nd(
                        targ_distributed_next_q_probabilities,
                        batch_indexed_next_qt_argmax)

                    # size of rewards: batch_size x 1
                    rewards = tf.expand_dims(single_device_rew_ph[i], -1)

                    # size of tiled_support: batch_size x num_atoms
                    tiled_support = tf.tile(self._support, [single_gpu_batch_size])
                    tiled_support = tf.reshape(tiled_support, [single_gpu_batch_size, self.num_atoms])

                    # size of target_support: batch_size x num_atoms

                    is_terminal_multiplier = 1. - tf.cast(single_device_done_ph[i], tf.float32)
                    # Incorporate terminal state to discount factor.
                    # size of gamma_with_terminal: batch_size x 1
                    gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
                    gamma_with_terminal = tf.expand_dims(gamma_with_terminal, -1)
                    target_support = rewards + gamma_with_terminal * tiled_support
                    self.check_list.append(target_support)
                    target_distribution, clipped_quotient, inner_prod = self._project_distribution(next_probabilities, target_support, self._support)
                    target_distribution = tf.stop_gradient(target_distribution)
                    self.check_list.append(target_distribution)
                    # size of indices: batch_size x 1.
                    indices = tf.expand_dims(tf.range(single_gpu_batch_size), -1)
                    # size of reshaped_actions: batch_size x 2.
                    reshaped_actions = tf.concat([indices, tf.expand_dims(single_device_act_ph[i], -1)], 1)
                    # For each element of the batch, fetch the logits for its selected action.
                    chosen_action_logits = tf.gather_nd(distributed_q_logits,
                                                        reshaped_actions)
                    chosen_action_probabilities = tf.gather_nd(distributed_q_probabilities,
                                                        reshaped_actions)                                
                    self.check_list.append(reshaped_actions)
                    self.check_list.append(chosen_action_logits)
                    self.check_list.append(clipped_quotient)
                    self.check_list.append(self._support)
                    self.check_list.append(next_probabilities)
                    self.check_list.append(eval_next_q_values)
                    self.check_list.append(next_qt_argmax)
                    self.check_list.append(inner_prod)
                    # tf.reduce_sum(chosen_action_logits * tf.log(chosen_action_logits + 1e-10), axis = -1)
                    loss =  - tf.reduce_sum(target_distribution * tf.log(chosen_action_probabilities + 1e-10), axis = -1)
                    # loss =  tf.reduce_mean(chosen_action_logits) - tf.reduce_mean(target_distribution)
                    # loss = tf.nn.softmax_cross_entropy_with_logits(
                        # labels=target_distribution,
                        # logits=chosen_action_logits)
                    self.check_list.append(loss)
                    self.train_params = tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, scope='main/eval')

                    losses.append(loss)
                    if is_gpu:
                        with tf.device('/gpu:{}'.format(gpu_list[i])):
                            grads.append(self.optimizer.compute_gradients(loss, self.train_params))
                    self.td_error = loss     
        if is_gpu:
            grad = self.average_gradients(grads)
            grad = self.clip_grad(grad)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope = 'main/eval')
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.apply_gradients(grad)

        self.loss = self.average_loss(losses)
        self.eval_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='main/eval')
        self.targ_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='main/targ')  
        self.targ_update_ops = [tf.assign(t, (1 - TAU) * e + TAU *t)
                                            for e,t in zip(self.eval_params, self.targ_params)]

        self.saver = tf.train.Saver(var_list=tf.global_variables())
        self.sess.run(tf.global_variables_initializer())

    def _project_distribution(self, next_probabilities, target_support, base_support,
                            validate_args=False):
        """Projects a batch of (support, next_probabilities) onto target_support.

        Based on equation (7) in (Bellemare et al., 2017):
            https://arxiv.org/abs/1707.06887
        In the rest of the comments we will refer to this equation simply as Eq7.

        This code is not easy to digest, so we will use a running example to clarify
        what is going on, with the following sample inputs:

            * target_support =       [[0, 2, 4, 6, 8],
                                [1, 3, 4, 5, 6]]
            * next_probabilities =        [[0.1, 0.6, 0.1, 0.1, 0.1],
                                [0.1, 0.2, 0.5, 0.1, 0.1]]
            * base_support = [4, 5, 6, 7, 8]

        In the code below, comments preceded with 'Ex:' will be referencing the above
        values.

        Args:
            target_support: Tensor of shape (batch_size, num_dims) defining target_support for the
            distribution.
            next_probabilities: Tensor of shape (batch_size, num_dims) defining next_probabilities on the
            original support points. Although for the CategoricalDQN agent these
            next_probabilities are probabilities, it is not required that they are.
            base_support: Tensor of shape (num_dims) defining support of the projected
            distribution. The values must be monotonically increasing. Vmin and Vmax
            will be inferred from the first and last elements of this tensor,
            respectively. The values in this tensor must be equally spaced.
            validate_args: Whether we will verify the contents of the
            base_support parameter.

        Returns:
            A Tensor of shape (batch_size, num_dims) with the projection of a batch of
            (support, next_probabilities) onto base_support.

        Raises:
            ValueError: If base_support has no dimensions, or if shapes of target_support,
            next_probabilities, and base_support are incompatible.
        """
        base_support_deltas = base_support[1:] - base_support[:-1]
        # delta_z = `\Delta z` in Eq7.
        delta_z = base_support_deltas[0]
        validate_deps = []
        target_support.shape.assert_is_compatible_with(next_probabilities.shape)
        target_support[0].shape.assert_is_compatible_with(base_support.shape)
        base_support.shape.assert_has_rank(1)
        if validate_args:
            # Assert that target_support and next_probabilities have the same shapes.
            validate_deps.append(
                tf.Assert(
                    tf.reduce_all(tf.equal(tf.shape(target_support), tf.shape(next_probabilities))),
                    [target_support, next_probabilities]))
            # Assert that elements of target_support and base_support have the same shape.
            validate_deps.append(
                tf.Assert(
                    tf.reduce_all(
                        tf.equal(tf.shape(target_support)[1], tf.shape(base_support))),
                    [target_support, base_support]))
            # Assert that base_support has a single dimension.
            validate_deps.append(
                tf.Assert(
                    tf.equal(tf.size(tf.shape(base_support)), 1), [base_support]))
            # Assert that the base_support is monotonically increasing.
            validate_deps.append(
                tf.Assert(tf.reduce_all(base_support_deltas > 0), [base_support]))
            # Assert that the values in base_support are equally spaced.
            validate_deps.append(
                tf.Assert(
                    tf.reduce_all(tf.equal(base_support_deltas, delta_z)),
                    [base_support]))

        with tf.control_dependencies(validate_deps):
            # Ex: `v_min, v_max = 4, 8`.
            v_min, v_max = base_support[0], base_support[-1]
            # Ex: `batch_size = 2`.
            batch_size = tf.shape(target_support)[0]
            # `N` in Eq7.
            # Ex: `num_dims = 5`.
            num_dims = tf.shape(base_support)[0]
            # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
            # Ex: `clipped_support = [[[ 4.  4.  4.  6.  8.]]
            #                         [[ 4.  4.  4.  5.  6.]]]`.
            clipped_support = tf.clip_by_value(target_support, v_min, v_max)[:, None, :]
            # Ex: `tiled_support = [[[[ 4.  4.  4.  6.  8.]
            #                         [ 4.  4.  4.  6.  8.]
            #                         [ 4.  4.  4.  6.  8.]
            #                         [ 4.  4.  4.  6.  8.]
            #                         [ 4.  4.  4.  6.  8.]]
            #                        [[ 4.  4.  4.  5.  6.]
            #                         [ 4.  4.  4.  5.  6.]
            #                         [ 4.  4.  4.  5.  6.]
            #                         [ 4.  4.  4.  5.  6.]
            #                         [ 4.  4.  4.  5.  6.]]]]`.
            tiled_support = tf.tile([clipped_support], [1, 1, num_dims, 1])
            # Ex: `reshaped_base_support = [[[ 4.]
            #                                  [ 5.]
            #                                  [ 6.]
            #                                  [ 7.]
            #                                  [ 8.]]
            #                                 [[ 4.]
            #                                  [ 5.]
            #                                  [ 6.]
            #                                  [ 7.]
            #                                  [ 8.]]]`.
            reshaped_base_support = tf.tile(tf.expand_dims(base_support, -1), [batch_size, 1])
            reshaped_base_support = tf.reshape(reshaped_base_support,
                                                [batch_size, num_dims, 1])
            # numerator = `|clipped_support - z_i|` in Eq7.
            # Ex: `numerator = [[[[ 0.  0.  0.  2.  4.]
            #                     [ 1.  1.  1.  1.  3.]
            #                     [ 2.  2.  2.  0.  2.]
            #                     [ 3.  3.  3.  1.  1.]
            #                     [ 4.  4.  4.  2.  0.]]
            #                    [[ 0.  0.  0.  1.  2.]
            #                     [ 1.  1.  1.  0.  1.]
            #                     [ 2.  2.  2.  1.  0.]
            #                     [ 3.  3.  3.  2.  1.]
            #                     [ 4.  4.  4.  3.  2.]]]]`.
            numerator = tf.abs(tiled_support - reshaped_base_support)
            quotient = 1 - (numerator / delta_z)
            # clipped_quotient = `[1 - numerator / (\Delta z)]_0^1` in Eq7.
            # Ex: `clipped_quotient = [[[[ 1.  1.  1.  0.  0.]     这一行的实际意义为，计算最终的标签分布的第一个支撑，
            #                            [ 0.  0.  0.  0.  0.]     它的概率值为来自target网络输出的第1个支撑的概率乘以1，
            #                            [ 0.  0.  0.  1.  0.]     第2个支撑的概率乘以1
            #                            [ 0.  0.  0.  0.  0.]     第3个支撑的概率乘以1
            #                            [ 0.  0.  0.  0.  1.]]    第4个支撑的概率乘以0
            #                           [[ 1.  1.  1.  0.  0.]     第5个支撑的概率乘以0
            #                            [ 0.  0.  0.  1.  0.]     既计算target网络的分布映射到标签分布第一个支撑的概率
            #                            [ 0.  0.  0.  0.  1.]     注意，只有距离小于1（delta_z）的概率会被传递，既概率的分配
            #                            [ 0.  0.  0.  0.  0.]
            #                            [ 0.  0.  0.  0.  0.]]]]`.
            clipped_quotient = tf.clip_by_value(quotient, 0, 1)
            # Ex: `next_probabilities = [[ 0.1  0.6  0.1  0.1  0.1]
            #                 [ 0.1  0.2  0.5  0.1  0.1]]`.
            next_probabilities = next_probabilities[:, None, :]
            # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))`
            # in Eq7.
            # Ex: `inner_prod = [[[[ 0.1  0.6  0.1  0.  0. ] 
            #                      [ 0.   0.   0.   0.  0. ]
            #                      [ 0.   0.   0.   0.1 0. ]
            #                      [ 0.   0.   0.   0.  0. ]
            #                      [ 0.   0.   0.   0.  0.1]]
            #                     [[ 0.1  0.2  0.5  0.  0. ]
            #                      [ 0.   0.   0.   0.1 0. ]
            #                      [ 0.   0.   0.   0.  0.1]
            #                      [ 0.   0.   0.   0.  0. ]
            #                      [ 0.   0.   0.   0.  0. ]]]]`.
            inner_prod = clipped_quotient * next_probabilities
            # Ex: `projection = [[ 0.8 0.0 0.1 0.0 0.1]
            #                    [ 0.8 0.1 0.1 0.0 0.0]]`.
            projection = tf.reduce_sum(inner_prod, 3)
            projection = tf.reshape(projection, [batch_size, num_dims])
            return projection, clipped_quotient, inner_prod

    def get_params(self):
        eval_params, targ_params = self.sess.run([self.eval_params, self.targ_params])
        return [eval_params, targ_params] 

    def pull_params(self, eval_params, targ_params):
        self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.eval_params, eval_params)]
        self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.targ_params, targ_params)]
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])



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
        
        # check_list = self.sess.run(self.check_list,{self.act_ph:act_ph,\
        #             self.rew_ph:rew_ph, self.done_ph:done_ph,\
        #             self.obs_ph:obs_ph, self.next_obs_ph:next_obs_ph,\
        #             self.eval_bn_flag:True, self.targ_bn_flag:True,\
        #             self.obs_ph:obs_ph, self.next_obs_ph:next_obs_ph}) 
        
    def check(self, obs_ph, next_obs_ph, act_ph, rew_ph, done_ph):

        
        check_list = self.sess.run(self.check_list,{self.act_ph:act_ph,\
                    self.rew_ph:rew_ph, self.done_ph:done_ph,\
                    self.obs_ph:obs_ph, self.next_obs_ph:next_obs_ph,\
                    self.eval_bn_flag:True, self.targ_bn_flag:True,\
                    self.obs_ph:obs_ph, self.next_obs_ph:next_obs_ph}) 

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
        actions = list()
        batch_size = len(obs_ph)
        predict_actions = self.sess.run(self._q_argmax, {self.obs_ph: obs_ph, self.eval_bn_flag:False, self.targ_bn_flag:False})
        epsilon = self.linearly_decaying_epsilon(
                            self.epsilon_decay_period,
                            self.training_steps,
                            self.min_replay_history,
                            self.epsilon_train)
        if self.training_steps%20000 == 0:
            print('----------epsilon-------------------------', epsilon)
        for i in range(batch_size):
            if is_eval:
                epsilon = self.epsilon_eval
            else:
                epsilon = self.linearly_decaying_epsilon(
                    self.epsilon_decay_period,
                    self.training_steps,
                    self.min_replay_history,
                    self.epsilon_train)
            if random.random() <= epsilon:
            # Choose a random action with probability epsilon.
                action = random.randint(0, self.a_dim - 1)
            else:
            # Choose the action with highest Q-value at the current state.
                if batch_size != 1:
                    action = predict_actions[i]
                else:
                    action = predict_actions
            actions.append(action)
        return actions
        
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
        
    def _build_net(self, obs, scope, bn_flag, trainable):
        init_w = tf.random_normal_initializer(0., 0.01)
        init_b = tf.constant_initializer(0.1)
        with tf.variable_scope(scope):
            x     = obs
            n_l1 = 512
            w1_s = tf.get_variable('w1_s', [x.shape[-1], n_l1], initializer=init_w, trainable=trainable)
            b1   = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w1_s) + b1
            x    = tf.nn.leaky_relu(x, alpha=0.2)
            
            n_l2 = 256
            w2_s = tf.get_variable('w2_s', [x.shape[-1], n_l2], initializer=init_w, trainable=trainable)
            b2   = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w2_s) + b2
            x    = tf.nn.leaky_relu(x, alpha=0.2)

            n_l3 = self.a_dim * self.num_atoms
            w3_s = tf.get_variable('w3_s', [x.shape[-1], n_l3], initializer=init_w, trainable=trainable)
            b3   = tf.get_variable('b3', [1, n_l3], initializer=init_b, trainable=trainable)
            x    = tf.matmul(x, w3_s) + b3
            x    = tf.reshape(x, (-1, self.a_dim, self.num_atoms))
            # x    = tf.nn.softmax(x)

            return x
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
    batch_size = 4
    agent = build_agent(4, 10, [2, 3], sess = tf.Session())
    # print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    # print(agent.execute_quantile_values)
    a = [2 for i in range(batch_size)]
    s = np.ones((batch_size, 10))
    s_ = np.ones((batch_size,10))*2
    r = np.ones(batch_size)*0.02
    done = np.zeros(batch_size)
    for i in range(5000):
        # r = r * (-1)
        agent.train(s, s_, a, r, done)
        if i %10==0:
            agent.soft_update()
    #     aaa = agent.get_action([[1,1 ,1]], True)
    #     q = agent.check_value([[1,1,1]])
        #print('qqq',q)