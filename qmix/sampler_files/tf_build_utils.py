import tensorflow as tf

def _build_central_c(self, hiden_states, actions, scope, trainable):
    # actions = [batch, num_agents]
    init_w = tf.random_normal_initializer(0., 0.01)
    init_b = tf.constant_initializer(0.01)
    with tf.variable_scope(scope):

        embedding_size = 20
        action_embeddings = tf.get_variable('action_embedding',
                                [self.a_dim, embedding_size])
        embedded_actions = tf.gather(action_embeddings, actions)
        print('embedded_actions', embedded_actions.shape)
        # embedded_actions = [batch, num_agents x embedding_size]
        reshape_embedded_actions = tf.reshape(embedded_actions, (-1, self.num_agents, embedding_size))
        

        x = tf.concat([reshape_embedded_actions, hiden_states], axis = -1)
        n_l1 = 256
        w1_s = tf.get_variable('w1_s', [x.shape[-1], n_l1], initializer=init_w, trainable=trainable)
        b1   = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
        x    = tf.matmul(x, w1_s) + b1
        x    = tf.nn.leaky_relu(x, alpha=0.2)


        

        n_l2 = 1
        w2_s = tf.get_variable('w2_s', [x.shape[-1], n_l2], initializer=init_w, trainable=trainable)
        b2   = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
        x    = tf.matmul(x, w2_s) + b2
        x    = tf.nn.leaky_relu(x, alpha=0.2)

        v_divide = tf.squeeze(x, axis = -1, name = 'v_divide')
        # n_hiden_state = hiden_states.shape[-1]
        # w_embed_act = tf.get_variable('w_embed_act', [reshape_embedded_actions.shape[-1], n_hiden_state], initializer=init_w, trainable=trainable)
        # b_embed_act = tf.get_variable('b_embed_act', [1, n_hiden_state], initializer=init_b, trainable=trainable)
        # hiden_embed = tf.matmul(reshape_embedded_actions, w_embed_act) + b_embed_act
        # hiden_embed = tf.nn.leaky_relu(hiden_embed, alpha=0.2)
        # print('hiden_embed', hiden_embed.shape)




        # 对于N个agent，复制N份hiden states
        # hiden_state = tf.reshape(hiden_state, (-1, 1, hiden_state.shape[-1]))
        # hiden_state = tf.tile(hiden_state, multiples=[1, self.num_agents, 1])

        # alliance_value = tf.multiply(hiden_embed, hiden_states)

        # v_divide = tf.reduce_sum(alliance_value, axis = -1)

        # v_tol = tf.reduce_sum(v_divide, axis = -1)

        return v_divide

def _build_dqn(hiden_state, obs1, obs2, a_dim, scope, trainable):
    init_w = tf.random_normal_initializer(0., 0.01)
    init_b = tf.constant_initializer(0.01)
    with tf.variable_scope(scope):
        x    = hiden_state
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

        n_l3 = a_dim
        w3_s = tf.get_variable('w3_s', [x.shape[-1], n_l3], initializer=init_w, trainable=trainable)
        b3   = tf.get_variable('b3', [1, n_l3], initializer=init_b, trainable=trainable)
        x    = tf.matmul(x, w3_s) + b3

        return x   

def _build_mix_hidden_state(self, cnn_out, global_state, action_masks, scope, trainable):
    init_w = tf.random_normal_initializer(0., 0.01)
    init_b = tf.constant_initializer(0.01)
    with tf.variable_scope(scope):

        x     = tf.concat([global_state, action_masks], axis = -1)
        n_s   = 1024
        w_s   = tf.get_variable('w_s', [x.shape[-1], n_s], initializer=init_w, trainable=trainable)
        b_s   = tf.get_variable('b_s', [1, n_s], initializer=init_b, trainable=trainable)
        x     = tf.matmul(x, w_s) + b_s
        x     = tf.nn.leaky_relu(x, alpha=0.2)
        x     = tf.concat([x, cnn_out], axis = -1)

        n_l1 = 512
        w1_s = tf.get_variable('w1_s', [x.shape[-1], n_l1], initializer=init_w, trainable=trainable)
        b1   = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
        x    = tf.matmul(x, w1_s) + b1
        x    = tf.nn.leaky_relu(x, alpha=0.2)

        return x
                            
def build_q_mix_v0(state, q, scope, trainable):
        init_w = tf.random_normal_initializer(0., 0.001)
        init_b = tf.random_normal_initializer(0., 0.001)
        with tf.variable_scope(scope):

            # state = tf.reshape(state, (-1, state.shape[-1] * state.shape[-2]))
            agent_num = q.shape[-1]

            # 这里之所以必须变成三维矩阵的原因在于，w1_s三维矩阵，而w1_s是三维矩阵的原因在于batch_size中每一个样本的s都是不同的，生成的权重自然也是不同的
            q     = tf.reshape(q, (-1, 1, agent_num))

            mixing_hidden_size = 512
            # 生成第一层权重矩阵w1_s
            w1_s   = tf.get_variable('w1_s', [state.shape[-1], agent_num * mixing_hidden_size], initializer=init_w, trainable=trainable)
            mix_s1 = tf.abs(tf.matmul(state, w1_s))
            mix_s1 = tf.reshape(mix_s1,(-1, agent_num, mixing_hidden_size))

            # 生成权重偏置w1_b
            w1_b   = tf.get_variable('w1_b', [state.shape[-1], mixing_hidden_size], initializer=init_b, trainable=trainable)
            mix_b1 = tf.matmul(state, w1_b)
            mix_b1 = tf.reshape(mix_b1,(-1, 1, mixing_hidden_size))

            x      = tf.nn.leaky_relu(tf.matmul(q, mix_s1) + mix_b1)
            # 这里的三维矩阵相乘 q = (B, 1, num_agent) X mix_s1 = (B, num_agent, mixing_hidden_size)
            # 计算顺序为，先将第一维度B忽略，计算 (1, num_agent) X (num_agent, mixing_hidden_size) = (1, mixing_hidden_size)
            # 所以最终的结果便为 (B, 1, mixing_hidden_size)
            #这里实现mix网络的第一层，q 与 W1相乘
            

            w2_s   = tf.get_variable('w2_s', [state.shape[-1], mixing_hidden_size], initializer=init_w, trainable=trainable)
            mix_s2 = tf.abs(tf.matmul(state, w2_s))
            mix_s2 = tf.reshape(mix_s2,(-1,mixing_hidden_size,1))

            w2_b_1 = tf.get_variable('w2_b_1', [state.shape[-1], mixing_hidden_size], initializer=init_b, trainable=trainable)
            s_b2_1 = tf.nn.leaky_relu(tf.matmul(state, w2_b_1))
            w2_b_2 = tf.get_variable('w2_b_2', [s_b2_1.shape[-1], 1], initializer=init_b, trainable=trainable)
            mix_b2 = tf.matmul(s_b2_1, w2_b_2)
            mix_b2 = tf.reshape(mix_b2,(-1, 1, 1))

            x      = tf.matmul(x, mix_s2) + mix_b2


            # x      = tf.matmul(x, mix_s2) + b2
            x      = tf.squeeze(x, axis=-1)
            x      = tf.squeeze(x, axis=-1)
            return x

def build_q_mix_v1(state, q, scope, trainable):
        init_w = tf.random_normal_initializer(0., 0.001)
        init_b = tf.constant_initializer(0.001)
        with tf.variable_scope(scope):

            # state = tf.reshape(state, (-1, state.shape[-1] * state.shape[-2]))
            agent_num = q.shape[-1]

            # 这里之所以必须变成三维矩阵的原因在于，w1_s三维矩阵，而w1_s是三维矩阵的原因在于batch_size中每一个样本的s都是不同的，生成的权重自然也是不同的
            q     = tf.reshape(q, (-1, 1, agent_num))

            mixing_hidden_size = 512
            # 生成第一层权重矩阵w1_s
            w1_s   = tf.get_variable('w1_s', [state.shape[-1], agent_num * mixing_hidden_size], initializer=init_w, trainable=trainable)
            mix_s1 = tf.abs(tf.matmul(state, w1_s))
            mix_s1 = tf.reshape(mix_s1,(-1, agent_num, mixing_hidden_size))

            # 生成权重偏置w1_b
            w1_b   = tf.get_variable('w1_b', [state.shape[-1], mixing_hidden_size], initializer=init_b, trainable=trainable)
            mix_b1 = tf.matmul(state, w1_b)
            mix_b1 = tf.reshape(mix_b1,(-1, 1, mixing_hidden_size))

            x      = tf.nn.leaky_relu(tf.matmul(q, mix_s1) + mix_b1)

            print('q', q.shape)
            print('x1', x.shape)
            x      = tf.squeeze(x, axis=1)
            print('x2', x.shape)
            
            w0     = tf.get_variable('w0', [x.shape[-1], 1], initializer=init_w, trainable=trainable)
            x      = tf.matmul(x, w0)
            x      = tf.squeeze(x, axis=-1)
            print('x3', x.shape)
            return x

def build_cpu_mix(network_functions, image, obs1, obs2, next_image, next_obs1, next_obs2, actions, r, done, a_dim, num_agents, mix_version):
    build_cnn = network_functions['build_cnn']
    build_dqn = network_functions['build_dqn']
    build_q_mix = network_functions['build_q_mix']
    build_hiden_state = network_functions['build_hiden_state']
    eval_bn = network_functions['eval_bn']
    targ_bn = network_functions['targ_bn']
    with tf.device('/cpu:0'):
        with tf.variable_scope('eval', reuse=tf.AUTO_REUSE):
            cpu_cnn_out_agent_1  = build_cnn(image[:, 0] ,'conv', eval_bn, True)  
            cpu_cnn_out_agent_2  = build_cnn(image[:, 1] ,'conv', eval_bn, True) 
            cpu_hiden_state_agent_1 = build_hiden_state(cpu_cnn_out_agent_1, obs1[:, 0], obs2[:, 0], 'hidden', eval_bn, True)
            cpu_hiden_state_agent_2 = build_hiden_state(cpu_cnn_out_agent_2, obs1[:, 1], obs2[:, 1], 'hidden', eval_bn, True)
            cpu_eval_q1 = build_dqn(cpu_hiden_state_agent_1, 'dqn', eval_bn, True)
            cpu_eval_q2 = build_dqn(cpu_hiden_state_agent_2, 'dqn', eval_bn, True)

            cpu_eval_next_cnn_out_agent_1  = build_cnn(next_image[:, 0] ,'conv', eval_bn, True)  
            cpu_eval_next_cnn_out_agent_2  = build_cnn(next_image[:, 1] ,'conv', eval_bn, True) 
            cpu_eval_next_hiden_state_agent_1 = build_hiden_state(cpu_eval_next_cnn_out_agent_1, next_obs1[:, 0], next_obs2[:, 0], 'hidden', eval_bn, True)
            cpu_eval_next_hiden_state_agent_2 = build_hiden_state(cpu_eval_next_cnn_out_agent_2, next_obs1[:, 1], next_obs2[:, 1], 'hidden', eval_bn, True)
            cpu_next_eval_q1 = build_dqn(cpu_eval_next_hiden_state_agent_1, 'dqn', eval_bn, True)
            cpu_next_eval_q2 = build_dqn(cpu_eval_next_hiden_state_agent_2, 'dqn', eval_bn, True)
            next_action1 = tf.argmax(cpu_next_eval_q1, axis = 1)
            next_action2 = tf.argmax(cpu_next_eval_q2, axis = 1)


        with tf.variable_scope('target', reuse=tf.AUTO_REUSE):
            cpu_next_cnn_out_agent_1  = build_cnn(next_image[:, 0] ,'conv', targ_bn, False)  
            cpu_next_cnn_out_agent_2  = build_cnn(next_image[:, 1] ,'conv', targ_bn, False) 
            cpu_next_hiden_state_agent_1 = build_hiden_state(cpu_next_cnn_out_agent_1, next_obs1[:, 0], next_obs2[:, 0], 'hidden', targ_bn, False)
            cpu_next_hiden_state_agent_2 = build_hiden_state(cpu_next_cnn_out_agent_2, next_obs1[:, 1], next_obs2[:, 1], 'hidden', targ_bn, False)
            cpu_target_q1 = build_dqn(cpu_next_hiden_state_agent_1, 'dqn', targ_bn, False)
            cpu_target_q2 = build_dqn(cpu_next_hiden_state_agent_2, 'dqn', targ_bn, False)
            # cpu_target_q1 = cpu_target_q1 - tf.math.abs(tf.random_normal(tf.shape(cpu_target_q1), stddev = 0.2))
            # cpu_target_q2 = cpu_target_q2 - tf.math.abs(tf.random_normal(tf.shape(cpu_target_q2), stddev = 0.2))

            cpu_sample_target_q1 = tf.reduce_sum(tf.one_hot(next_action1, depth=a_dim) *cpu_target_q1, axis=1)
            cpu_sample_target_q2 = tf.reduce_sum(tf.one_hot(next_action2, depth=a_dim) *cpu_target_q2, axis=1)
            reshaped_cpu_sample_target_q1 = tf.reshape(cpu_sample_target_q1, (-1, 1))
            reshaped_cpu_sample_target_q2 = tf.reshape(cpu_sample_target_q2, (-1, 1))



        with tf.variable_scope('mixer', reuse=tf.AUTO_REUSE):
            cpu_sampled_q1 = tf.reshape(tf.reduce_sum(tf.one_hot(actions[:, 0], depth=a_dim) *cpu_eval_q1,axis=1), (-1, 1))
            cpu_sampled_q2 = tf.reshape(tf.reduce_sum(tf.one_hot(actions[:, 1], depth=a_dim) *cpu_eval_q2,axis=1), (-1, 1))
            
            cpu_eval_q_input  = tf.concat([cpu_sampled_q1 , cpu_sampled_q2], axis=1)
            cpu_eval_q_input  = tf.reshape(cpu_eval_q_input,(-1, 1, num_agents))

            cpu_target_q_input  = tf.concat([reshaped_cpu_sample_target_q1 , reshaped_cpu_sample_target_q2], axis=1)
            cpu_target_q_input  = tf.reshape(cpu_target_q_input,(-1, 1, num_agents))



            cpu_mix_cnn_out1 = build_cnn(image[:, 0] ,'conv', eval_bn, True)
            cpu_mix_cnn_out2 = build_cnn(image[:, 1] ,'conv', eval_bn, True)
            cpu_mix_hiden_state = tf.concat([
                cpu_mix_cnn_out1, cpu_mix_cnn_out2, obs1[:, 0], obs2[:, 0], obs2[:, 1]],\
                axis = -1)

            cpu_mix_next_cnn_out1 = build_cnn(next_image[:, 0] ,'conv', eval_bn, True)
            cpu_mix_next_cnn_out2 = build_cnn(next_image[:, 1] ,'conv', eval_bn, True)
            cpu_mix_next_hiden_state = tf.concat([
                cpu_mix_next_cnn_out1, cpu_mix_next_cnn_out2, next_obs1[:, 0], next_obs2[:, 0], next_obs2[:, 1]], \
                axis = -1)

            cpu_mix_next_hiden_state = tf.stop_gradient(cpu_mix_next_hiden_state)
            cpu_eval_q_total   = build_q_mix(cpu_mix_hiden_state, cpu_eval_q_input ,'eval', True)
            cpu_target_q_total = build_q_mix(cpu_mix_next_hiden_state, cpu_target_q_input,'target', False)

            cpu_q_target = r + 0.99 * (1- done) * cpu_target_q_total
            td_error     = (cpu_q_target- cpu_eval_q_total)**2*0.5
    return td_error, cpu_eval_q_total, cpu_target_q_total, cpu_eval_q1, cpu_eval_q2, cpu_hiden_state_agent_2