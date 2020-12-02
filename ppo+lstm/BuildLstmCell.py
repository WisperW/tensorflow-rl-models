import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
from sampler_files.tf_utils import ortho_init, batch_norm
def _ln(x, g, b, e=1e-5, axes=[0]):
    u, s = tf.nn.moments(x, axes=axes, keep_dims=True)
    x = (x-u)/tf.sqrt(s+e)
    x = x*g+b
    return x

class LSTMCell():
    def __init__(self, lstm_dim, scope, keep_prob, init_scale=1.0):
        self.lstm_dim = lstm_dim
        self.scope = scope
        self.init_scale = init_scale

    def __call__(self, inputx, current_state, new_traj_start):
        with tf.variable_scope(self.scope):
            wx = tf.get_variable("wx", [inputx.shape[-1], self.lstm_dim*4], initializer=ortho_init(scale=1.0))
            wh = tf.get_variable("wh", [self.lstm_dim, self.lstm_dim*4], initializer=ortho_init(scale=1.0))
            b = tf.get_variable("b", [self.lstm_dim*4], initializer=tf.constant_initializer(0.0))

        c, h = tf.split(axis=1, num_or_size_splits=2, value=current_state)
        c = tf.where(tf.cast(new_traj_start, dtype=tf.bool), tf.zeros_like(c), c)
        h = tf.where(tf.cast(new_traj_start, dtype=tf.bool), tf.zeros_like(h), h)
        z = tf.matmul(inputx, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(c)

        current_state = tf.concat(axis=1, values=[c, h])
        output = h
        check_value = tf.matmul(inputx, wx) + tf.matmul(h, wh)
        return output, current_state, check_value

class DropoutLSTMCell():
    def __init__(self, lstm_dim, scope, keep_prob, init_scale=1.0):
        self.lstm_dim = lstm_dim
        self.scope = scope
        self.keep_prob = keep_prob
        self.init_scale = init_scale

    def __call__(self, inputx, current_state, new_traj_start):
        with tf.variable_scope(self.scope):
            wx = tf.get_variable("wx", [inputx.shape[-1], self.lstm_dim*4], initializer=tf.orthogonal_initializer)
            wh = tf.get_variable("wh", [self.lstm_dim, self.lstm_dim*4], initializer=tf.orthogonal_initializer)
            b = tf.get_variable("b", [self.lstm_dim*4], initializer=tf.constant_initializer(0.0))

        c, h = tf.split(axis=1, num_or_size_splits=2, value=current_state)
        c = tf.where(tf.cast(new_traj_start, dtype=tf.bool), tf.zeros_like(c), c)
        h = tf.where(tf.cast(new_traj_start, dtype=tf.bool), tf.zeros_like(h), h)
        z = tf.matmul(inputx, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(c)
        h = tf.nn.dropout(h, self.keep_prob)
        current_state = tf.concat(axis=1, values=[c, h])
        output = h
        check_value = tf.matmul(inputx, wx) + tf.matmul(h, wh)
        return output, current_state, check_value

class LNLSTMCell():
    def __init__(self, lstm_dim, scope, init_scale=1.0):
        self.lstm_dim = lstm_dim
        self.scope = scope
        self.init_scale = init_scale

    def __call__(self, inputx, current_state, new_traj_start):
        with tf.variable_scope(self.scope):

            wx = tf.get_variable("wx", [inputx.shape[-1], self.lstm_dim*4], initializer=tf.orthogonal_initializer)
            gx = tf.get_variable("gx", [self.lstm_dim*4], initializer=tf.constant_initializer(1.0))
            bx = tf.get_variable("bx", [self.lstm_dim*4], initializer=tf.constant_initializer(0.0))

            wh = tf.get_variable("wh", [self.lstm_dim, self.lstm_dim*4], initializer=tf.orthogonal_initializer)
            gh = tf.get_variable("gh", [self.lstm_dim*4], initializer=tf.constant_initializer(1.0))
            bh = tf.get_variable("bh", [self.lstm_dim*4], initializer=tf.constant_initializer(0.0))

            b = tf.get_variable("b", [self.lstm_dim*4], initializer=tf.constant_initializer(0.0))

            gc = tf.get_variable("gc", [self.lstm_dim], initializer=tf.constant_initializer(1.0))
            bc = tf.get_variable("bc", [self.lstm_dim], initializer=tf.constant_initializer(0.0))

        c, h = tf.split(axis=1, num_or_size_splits=2, value=current_state)
        c = tf.where(tf.cast(new_traj_start, dtype=tf.bool), tf.zeros_like(c), c)
        h = tf.where(tf.cast(new_traj_start, dtype=tf.bool), tf.zeros_like(h), h)
        z = _ln(tf.matmul(inputx, wx), gx, bx) + _ln(tf.matmul(h, wh), gh, bh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(_ln(c, gc, bc))

        current_state = tf.concat(axis=1, values=[c, h])
        output = h
        check_value = tf.matmul(inputx, wx) + tf.matmul(h, wh)
        return output, current_state, check_value

def dynamic_lstm(lstm_cell, inputx_sequence, lstm_init_state, new_traj_starts, is_single_time_step = False):
    # shape of inputx is (batchsize, time_step, ...)
    # shape of new_traj_starts is (batchsize, time_step)
    input_shape = array_ops.shape(inputx_sequence)
    batchsize   = input_shape[0]
    traj_length = input_shape[1]

    outputs_array = tf.TensorArray(dtype = tf.float32, size=1, dynamic_size=True)
    output_hidden_states_array = tf.TensorArray(dtype = tf.float32, size=1, dynamic_size=True)
    check_array = tf.TensorArray(dtype = tf.float32, size=1, dynamic_size=True)
    current_state = lstm_init_state

    def cond(traj_ptr, traj_length, current_state, inputx_sequence, new_traj_starts, outputs_array, output_hidden_states_array, check_array):
        return traj_ptr < traj_length
    
    def body(traj_ptr, traj_length, current_state, inputx_sequence, new_traj_starts, outputs_array, output_hidden_states_array, check_array):

        inputx = inputx_sequence[:, traj_ptr]
        new_traj_start = new_traj_starts[:, traj_ptr]
        output, current_state, check_value = lstm_cell(inputx, current_state, new_traj_start)
        
        outputs_array = outputs_array.write(traj_ptr, [output])
        output_hidden_states_array = output_hidden_states_array.write(traj_ptr, [current_state])
        check_array = check_array.write(traj_ptr, [check_value])
        traj_ptr += 1


        return traj_ptr, traj_length, current_state, inputx_sequence, new_traj_starts, outputs_array, output_hidden_states_array, check_array

    traj_ptr = tf.constant(0)
    traj_ptr, traj_length, current_state, inputx_sequence, new_traj_starts, outputs_array, output_hidden_states_array, check_array  = tf.while_loop(cond, body, 
                                                        [traj_ptr, 
                                                         traj_length, 
                                                         current_state, 
                                                         inputx_sequence, 
                                                         new_traj_starts, 
                                                         outputs_array,
                                                         output_hidden_states_array,
                                                         check_array
                                                         ]) 

    outputs = outputs_array.stack()
    outputs = tf.squeeze(outputs)
    if not is_single_time_step:
        outputs = tf.transpose(outputs, [1, 0, 2])
    # (time_step, batchsize, lstm_dim) to (batchsize, time_step, lstm_dim)

    output_hidden_states = output_hidden_states_array.stack()
    output_hidden_states = tf.squeeze(output_hidden_states)
    if not is_single_time_step:
        output_hidden_states = tf.transpose(output_hidden_states, [1, 0, 2])

    check_values = check_array.stack()
    check_values = tf.squeeze(check_values)
    if not is_single_time_step:
        check_values = tf.transpose(check_values, [1, 0, 2])
    return outputs, output_hidden_states, check_values