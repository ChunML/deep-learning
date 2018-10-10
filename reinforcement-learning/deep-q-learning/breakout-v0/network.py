import tensorflow as tf
from functools import reduce


class QNetwork:
    def __init__(self, height, width, channel,
                 learning_rate=0.01, state_size=4,
                 action_size=4,
                 name='QNetwork'):
        with tf.variable_scope(name):
            # states as network input
            self.inputs_ = tf.placeholder(
                tf.float32, [None, height, width, channel], name='inputs')

            # chosen actions to compute Q values
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)

            with tf.variable_scope('conv1'):
                W = tf.get_variable('weights', [8, 8, 3, 16],
                                    initializer=tf.truncated_normal_initializer(stddev=5e-2))
                b = tf.get_variable(
                    'biases', [16], initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(self.inputs_, W, strides=[
                                    1, 4, 4, 1], padding='VALID')
                self.conv1 = tf.nn.relu(tf.nn.bias_add(conv, b), name='conv1')

            with tf.variable_scope('conv2'):
                W = tf.get_variable('weights', [4, 4, 16, 32],
                                    initializer=tf.truncated_normal_initializer(stddev=5e-2))
                b = tf.get_variable(
                    'biases', [32], initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(self.conv1, W, strides=[
                                    1, 2, 2, 1], padding='VALID')
                self.conv2 = tf.nn.relu(tf.nn.bias_add(conv, b), name='conv2')

            with tf.variable_scope('fc'):
                s = self.conv2.get_shape().as_list()
                #reshape = tf.reshape(self.conv3, [-1, s[1] * s[2] * s[3]])
                reshape = tf.reshape(
                    self.conv2, [-1, reduce(lambda x, y: x * y, s[1:])])
                dim = reshape.get_shape()[1].value
                W = tf.get_variable('weights', [dim, 256],
                                    initializer=tf.truncated_normal_initializer(stddev=4e-2))
                b = tf.get_variable(
                    'biases', [256], initializer=tf.constant_initializer(0.1))
                self.fc = tf.nn.relu(tf.matmul(reshape, W) + b, name='fc1')

            # Target Q values for training
            self.targetQs_ = tf.placeholder(
                tf.float32, [None], name='target_Q')

            # Output layer
            self.output = tf.contrib.layers.fully_connected(self.fc, action_size,
                                                            activation_fn=None)

            # Q values from network
            self.Q = tf.reduce_sum(tf.multiply(
                self.output, one_hot_actions), axis=1)

            # Compute training loss
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))

            # Training opt
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate).minimize(self.loss)
