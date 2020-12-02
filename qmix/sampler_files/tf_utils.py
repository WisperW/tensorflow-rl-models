import tensorflow as tf
import numpy as np
def get_logp_action_mask(action, logits, action_masks):
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
    #       the implementation does not allow second-order derivatives...
    one_hot_actions = tf.one_hot(action, logits.get_shape().as_list()[-1])
    one_hot_actions = tf.stop_gradient(one_hot_actions)
    
    # Prevent invalid actions backpropagation
    logits = tf.multiply(logits, action_masks)
    ''' 当logits中的值小于-36时，超出tensorflow的计算精度，使用softmax计算会得到0的结果，如果此时的采样action为其对应值，则会导致nan，因此对logits的值加以限制'''
    logits = tf.clip_by_value(logits, -30, 10000)
    # bug exp
    # logits [ -0.89203244   5.798719    -1.3013482  -38.33122    -11.059391
    #          -9.4196005   50.80005     -0.           0.           0.
    #          -6.296147    10.2524395 ]

    # Calculate softmax and correct the invalid action probability to 0
    softmax = tf.nn.softmax(logits)
    # bug exp
    # softmax [3.5515200e-23 2.8587100e-20 2.3585775e-23 0.0000000e+00 1.3639129e-27
    #          7.0297229e-27 1.0000000e+00 8.6660088e-23 8.6660088e-23 8.6660088e-23
    #          1.5974876e-25 2.4569587e-18]


    # nan_prevention = tf.constant(1e-5)
    # softmax = tf.add(softmax, nan_prevention)
    # debug = softmax
    exp_logits = softmax * tf.reduce_sum(tf.exp(logits), axis=-1, keepdims=True)
    exp_logits = tf.multiply(exp_logits, action_masks)
    softmax = exp_logits / tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
    pi = tf.reduce_sum(tf.multiply(softmax, one_hot_actions), axis=-1)
    log_pi = tf.log(pi)
    return log_pi

def cat_entropy_action_mask(logits, action_masks):

    logits = tf.multiply(logits, action_masks)
    a_0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    exp_a_0 = tf.exp(a_0)
    exp_a_0 = tf.multiply(exp_a_0, action_masks)
    z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
    p_0 = exp_a_0 / z_0
    return tf.reduce_sum(p_0 * (tf.log(z_0) - a_0), axis=-1)

def get_sample_op_action_mask(logits, action_masks, name):

    negative_inf_vector = tf.ones_like(action_masks, dtype=tf.float32) * -np.inf
    zero_vector = tf.zeros_like(action_masks, dtype=tf.float32)
    neg_inf_action_masks = tf.where(tf.cast(action_masks, dtype=tf.bool), zero_vector, negative_inf_vector)
    logits = tf.add(logits, neg_inf_action_masks)
    sample_logit_op = tf.squeeze(tf.multinomial(logits, 1), axis=1, name = name)
    return sample_logit_op

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init
    
def get_sample_op_action_mask_q_learning(logits, action_masks, name):

    negative_inf_vector = tf.ones_like(action_masks, dtype=tf.float32) * -np.inf
    zero_vector = tf.zeros_like(action_masks, dtype=tf.float32)
    neg_inf_action_masks = tf.where(tf.cast(action_masks, dtype=tf.bool), zero_vector, negative_inf_vector)
    logits = tf.add(logits, neg_inf_action_masks)
    # !!!!!!! pay attention to axis in tf.argmax
    sample_logit_op = tf.argmax(logits, axis= 1, name = name)
    return sample_logit_op

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None,
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
        

        # w = tf.get_variable("W", filter_shape, dtype,initializer=tf.random_uniform_initializer(-w_bound, w_bound),
        #                     collections=collections)
        # b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.random_uniform_initializer(-w_bound, w_bound),
        #                     collections=collections)

        w = tf.get_variable("W", filter_shape, dtype, tf.orthogonal_initializer(),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.orthogonal_initializer(),
                            collections=collections)

        if summary_tag is not None:
            tf.summary.image(summary_tag,
                            tf.transpose(tf.reshape(w, [filter_size[0], filter_size[1], -1, 1]),
                                        [2, 0, 1, 3]))

        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def ResBlock(x_input, channel, scope):
    x_res = x_input
    x = tf.nn.leaky_relu(x_input, alpha=0.2)
    x = conv2d(x, channel, scope, [3, 3], [1, 1], pad="SAME")
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = conv2d(x, channel, scope, [3, 3], [1, 1], pad="SAME")
    return tf.add(x_res, x)

def BuildBlock(x, out_channel, scope):
    with tf.variable_scope(scope):
        x = ResBlock(x, out_channel, 'block1')
        x = ResBlock(x, out_channel, 'block2')
        x = ResBlock(x, out_channel, 'block3')
        return x


def build_cpu_double_vdn(image, obs1, obs2, next_image, next_obs1, next_obs2, actions, r, done, a_dim, num_agents):
    with tf.device('/cpu:0'):
        with tf.variable_scope('eval', reuse=tf.AUTO_REUSE):
            cnn_out_agent_1  = build_cnn(image[:, 0] ,'conv', True)  
            cnn_out_agent_2  = build_cnn(image[:, 1] ,'conv', True) 
            cpu_hiden_state_agent_1 = build_hiden_state(cnn_out_agent_1, obs1[:, 0], obs2[:, 0], 'hidden', True)
            cpu_hiden_state_agent_2 = build_hiden_state(cnn_out_agent_2, obs1[:, 1], obs2[:, 1], 'hidden', True)
            cpu_eval_q1 = build_dqn(cpu_hiden_state_agent_1, obs1[:, 0], obs2[:, 0], a_dim, 'dqn', True)
            cpu_eval_q2 = build_dqn(cpu_hiden_state_agent_2, obs1[:, 1], obs2[:, 1], a_dim, 'dqn', True)

            cpu_eval_next_cnn_out_agent_1  = build_cnn(next_image[:, 0] ,'conv', True)  
            cpu_eval_next_cnn_out_agent_2  = build_cnn(next_image[:, 1] ,'conv', True) 
            cpu_eval_next_hiden_state_agent_1 = build_hiden_state(cpu_eval_next_cnn_out_agent_1, next_obs1[:, 0], next_obs2[:, 0], 'hidden', True)
            cpu_eval_next_hiden_state_agent_2 = build_hiden_state(cpu_eval_next_cnn_out_agent_2, next_obs1[:, 1], next_obs2[:, 1], 'hidden', True)
            cpu_next_eval_q1 = build_dqn(cpu_eval_next_hiden_state_agent_1, next_obs1[:, 0], next_obs2[:, 0], a_dim, 'dqn', True)
            cpu_next_eval_q2 = build_dqn(cpu_eval_next_hiden_state_agent_2, next_obs1[:, 1], next_obs2[:, 1], a_dim, 'dqn', True)
            next_action1 = tf.argmax(cpu_next_eval_q1, axis = 1)
            next_action2 = tf.argmax(cpu_next_eval_q2, axis = 1)

        with tf.variable_scope('target', reuse=tf.AUTO_REUSE):
            cpu_next_cnn_out_agent_1  = build_cnn(next_image[:, 0] ,'conv', False)  
            cpu_next_cnn_out_agent_2  = build_cnn(next_image[:, 1] ,'conv', False) 
            cpu_next_hiden_state_agent_1 = build_hiden_state(cpu_next_cnn_out_agent_1, next_obs1[:, 0], next_obs2[:, 0], 'hidden', False)
            cpu_next_hiden_state_agent_2 = build_hiden_state(cpu_next_cnn_out_agent_2, next_obs1[:, 1], next_obs2[:, 1], 'hidden', False)
            cpu_target_q1 = build_dqn(cpu_next_hiden_state_agent_1, next_obs1[:, 0], next_obs2[:, 0], a_dim, 'dqn', False)
            cpu_target_q2 = build_dqn(cpu_next_hiden_state_agent_2, next_obs1[:, 1], next_obs2[:, 1], a_dim, 'dqn', False)


            cpu_sample_target_q1 = tf.reduce_sum(tf.one_hot(next_action1, depth=a_dim) *cpu_target_q1, axis=1)
            cpu_sample_target_q2 = tf.reduce_sum(tf.one_hot(next_action2, depth=a_dim) *cpu_target_q2, axis=1)
            reshaped_cpu_sample_target_q1 = tf.reshape(cpu_sample_target_q1, (-1, 1))
            reshaped_cpu_sample_target_q2 = tf.reshape(cpu_sample_target_q2, (-1, 1))

        sampled_q1 = tf.reshape(tf.reduce_sum(tf.one_hot(actions[:, 0], depth=a_dim) *cpu_eval_q1,axis=1), (-1, 1))
        sampled_q2 = tf.reshape(tf.reduce_sum(tf.one_hot(actions[:, 1], depth=a_dim) *cpu_eval_q2,axis=1), (-1, 1))
        cpu_eval_q_total   = tf.squeeze((sampled_q1 + sampled_q2), axis=-1)
        cpu_target_q_total = tf.squeeze((reshaped_cpu_sample_target_q1 + reshaped_cpu_sample_target_q2), axis=-1)

        cpu_q_target = r + 0.99 * (1- done) * cpu_target_q_total
        td_error     = (cpu_q_target- cpu_eval_q_total)**2*0.5
        
    return td_error, cpu_eval_q_total, cpu_target_q_total, cpu_eval_q1, cpu_eval_q2

def build_cpu_vdn(image, obs1, obs2, next_image, next_obs1, next_obs2, actions, r, done, a_dim, num_agents):
    with tf.device('/cpu:0'):
        with tf.variable_scope('eval', reuse=tf.AUTO_REUSE):
            cnn_out_agent_1  = build_cnn(image[:, 0] ,'conv', True)  
            cnn_out_agent_2  = build_cnn(image[:, 1] ,'conv', True) 
            cpu_hiden_state_agent_1 = build_hiden_state(cnn_out_agent_1, obs1[:, 0], obs2[:, 0], 'hidden', True)
            cpu_hiden_state_agent_2 = build_hiden_state(cnn_out_agent_2, obs1[:, 1], obs2[:, 1], 'hidden', True)
            cpu_eval_q1 = build_dqn(cpu_hiden_state_agent_1, obs1[:, 0], obs2[:, 0], a_dim, 'dqn', True)
            cpu_eval_q2 = build_dqn(cpu_hiden_state_agent_2, obs1[:, 1], obs2[:, 1], a_dim, 'dqn', True)

        with tf.variable_scope('target', reuse=tf.AUTO_REUSE):
            cpu_next_cnn_out_agent_1  = build_cnn(next_image[:, 0] ,'conv', False)  
            cpu_next_cnn_out_agent_2  = build_cnn(next_image[:, 1] ,'conv', False) 
            cpu_next_hiden_state_agent_1 = build_hiden_state(cpu_next_cnn_out_agent_1, next_obs1[:, 0], next_obs2[:, 0], 'hidden', False)
            cpu_next_hiden_state_agent_2 = build_hiden_state(cpu_next_cnn_out_agent_2, next_obs1[:, 1], next_obs2[:, 1], 'hidden', False)
            cpu_target_q1 = build_dqn(cpu_next_hiden_state_agent_1, next_obs1[:, 0], next_obs2[:, 0], a_dim, 'dqn', False)
            cpu_target_q2 = build_dqn(cpu_next_hiden_state_agent_2, next_obs1[:, 1], next_obs2[:, 1], a_dim, 'dqn', False)
            reshaped_cpu_sample_target_q1 = tf.reshape(tf.reduce_max(cpu_target_q1, -1), (-1, 1))
            reshaped_cpu_sample_target_q2 = tf.reshape(tf.reduce_max(cpu_target_q2, -1), (-1, 1))

        sampled_q1 = tf.reshape(tf.reduce_sum(tf.one_hot(actions[:, 0], depth=a_dim) *cpu_eval_q1,axis=1), (-1, 1))
        sampled_q2 = tf.reshape(tf.reduce_sum(tf.one_hot(actions[:, 1], depth=a_dim) *cpu_eval_q2,axis=1), (-1, 1))
        cpu_eval_q_total   = tf.squeeze((sampled_q1 + sampled_q2), axis=-1)
        cpu_target_q_total = tf.squeeze((reshaped_cpu_sample_target_q1 + reshaped_cpu_sample_target_q2), axis=-1)

        cpu_q_target = r + 0.99 * (1- done) * cpu_target_q_total
        td_error     = (cpu_q_target- cpu_eval_q_total)**2*0.5
        
    return td_error, cpu_eval_q_total, cpu_target_q_total, cpu_eval_q1, cpu_eval_q2


