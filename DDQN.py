import tensorflow as tf
from keras.losses import logcosh
from keras.activations import relu
from keras.initializers import VarianceScaling
from keras.layers import Dense, Conv2D, Flatten


class DDQN:
    """ Implements a Dueling Dual Deep Q-Network based on the frames of the Retro Environment """
    def __init__(self, n_actions, hidden=1024, frame_height=82, frame_width=128, stacked_frames=4,
                 learning_rate=0.00001):
        self.n_actions = n_actions
        self.hidden = hidden
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.stacked_frames = stacked_frames
        self.learning_rate = learning_rate

        self.input = tf.placeholder(shape=[None, self.frame_height, self.frame_width, self.stacked_frames],
                                    dtype=tf.float32)
        self.input = self.input / 255

        # Convolutional layers
        self.conv1 = self.conv_layer(self.input, 32, [8, 8], 4, 'conv1')
        self.conv2 = self.conv_layer(self.conv1, 64, [4, 4], 2, 'conv2')
        self.conv3 = self.conv_layer(self.conv2, 64, [3, 3], 1, 'conv3')
        self.conv4 = self.conv_layer(self.conv3, self.hidden, [5, 5], 1, 'conv4')

        # Splitting into value and advantage streams
        self.v_stream, self.a_stream = tf.split(self.conv4, 2, 3)
        self.v_stream = Flatten()(self.v_stream)
        self.value = self.dense_layer(self.v_stream, 1, 'value')
        self.a_stream = Flatten()(self.a_stream)
        self.advantage = self.dense_layer(self.a_stream, self.n_actions, 'advantage')

        # Getting Q-values from value and advantage streams
        self.q_values = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
        self.prediction = tf.argmax(self.q_values, 1)

        # targetQ according to Bellman equation
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        self.action_one_hot = tf.one_hot(self.action, self.n_actions, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, self.action_one_hot), axis=1)

        # Parameter updates
        self.error = logcosh(self.target_q, self.Q)
        self.loss = tf.reduce_mean(self.error)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)

    @staticmethod
    def conv_layer(_inputs, _filters, _kernel_size, _strides, _name):
        return Conv2D(filters=_filters, kernel_size=_kernel_size, strides=_strides,
                      kernel_initializer=VarianceScaling(scale=2.0), padding="valid",
                      activation=relu, use_bias=False, name=_name)(_inputs)

    @staticmethod
    def dense_layer(_inputs, _units, _name):
        return Dense(units=_units, kernel_initializer=VarianceScaling(scale=2.0), name=_name)(_inputs)


# class DDQN:
#
#     def __init__(self, n_actions, hidden=1024, frame_height=82, frame_width=128, stacked_frames=4,
#                  learning_rate=0.00001):
#         self.n_actions = n_actions
#         self.hidden = hidden
#         self.frame_height = frame_height
#         self.frame_width = frame_width
#         self.stacked_frames = stacked_frames
#         self.learning_rate = learning_rate
#
#         self.input = tf.placeholder(shape=[None, self.frame_height, self.frame_width, self.stacked_frames],
#                                     dtype=tf.float32)
#         self.input = self.input / 255
#
#         # Convolutional layers
#         self.conv1 = self.conv_layer(self.input, 32, [8, 8], 4, 'conv1')
#         self.conv2 = self.conv_layer(self.conv1, 64, [4, 4], 2, 'conv2')
#         self.conv3 = self.conv_layer(self.conv2, 64, [3, 3], 1, 'conv3')
#         self.conv4 = self.conv_layer(self.conv3, self.hidden, [5, 5], 1, 'conv4')
#
#         # Splitting into value and advantage stream
#         self.v_stream, self.a_stream = tf.split(self.conv4, 2, 3)
#         self.v_stream = tf.layers.flatten(self.v_stream)
#         self.value = self.dense_layer(self.v_stream, 1, 'value')
#         self.a_stream = tf.layers.flatten(self.a_stream)
#         self.advantage = self.dense_layer(self.a_stream, self.n_actions, 'advantage')
#
#         # Combining value and advantage into Q-values
#         self.q_values = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
#         self.prediction = tf.argmax(self.q_values, 1)
#
#         # targetQ according to Bellman equation
#         self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
#         # Action that was performed
#         self.action = tf.placeholder(shape=[None], dtype=tf.int32)
#         self.action_one_hot = tf.one_hot(self.action, self.n_actions, dtype=tf.float32)
#         self.Q = tf.reduce_sum(tf.multiply(self.q_values, self.action_one_hot), axis=1)
#
#         # Parameter updates
#         self.error = tf.losses.huber_loss(labels=self.target_q, predictions=self.Q)
#         self.loss = tf.reduce_mean(self.error)
#         self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
#         self.update = self.optimizer.minimize(self.loss)
#
#     @staticmethod
#     def conv_layer(_inputs, _filters, _kernel_size, _strides, _name):
#         return tf.layers.conv2d(
#             inputs=_inputs, filters=_filters, kernel_size=_kernel_size, strides=_strides,
#             kernel_initializer=tf.variance_scaling_initializer(scale=2),
#             padding="valid", activation=tf.nn.relu, use_bias=False, name=_name)
#
#     @staticmethod
#     def dense_layer(_inputs, _units, _name):
#         return tf.layers.dense(
#             inputs=_inputs, units=_units,
#             kernel_initializer=tf.variance_scaling_initializer(scale=2), name=_name)


class TargetNetworkUpdater:

    """ Updates the variables and the weights of the target network based on the main network """
    def __init__(self, main_vars, target_vars):
        self.main_vars = main_vars
        self.target_vars = target_vars

    def update_target_vars(self):
        update_ops = []
        for i, var in enumerate(self.main_vars):
            copy_op = self.target_vars[i].assign(var.value())
            update_ops.append(copy_op)
        return update_ops

    def update_networks(self, sess):
        update_ops = self.update_target_vars()
        for copy_op in update_ops:
            sess.run(copy_op)
