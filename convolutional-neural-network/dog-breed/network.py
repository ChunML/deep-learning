import tensorflow as tf
import numpy as np
from functools import reduce
from time import time


class Network(object):
    # , decay_steps, decay_factor=0.94):
    def __init__(self, images, labels, ouput_size, initial_learning_rate, is_training=True):
        self.is_training = is_training
        # with tf.name_scope('conv1') as scope:
        pool1 = self._build_conv2d_block(images, 'conv1', 64, 2)
        # with tf.name_scope('conv2') as scope:
        pool2 = self._build_conv2d_block(pool1, 'conv2', 128, 2)
        # with tf.name_scope('conv3') as scope:
        pool3 = self._build_conv2d_block(pool2, 'conv3', 256, 3)
        # with tf.name_scope('conv4') as scope:
        pool4 = self._build_conv2d_block(pool3, 'conv4', 512, 3)
        # with tf.name_scope('conv5') as scope:
        pool5 = self._build_conv2d_block(pool4, 'conv5', 512, 3)

        # pool5_shape = pool5.get_shape().as_list()
        # flatten = tf.reshape(
        #     pool5, [-1, reduce(lambda x, y: x * y, pool5_shape[1:])])

        flatten = tf.layers.flatten(pool5)

        fc6 = self._fc(flatten, 'fc6', 2048)
        fc7 = self._fc(fc6, 'fc7', 1024)
        logits = self._fc(
            fc7, 'logits', ouput_size, activate_func=None)
        self.predictions = tf.argmax(logits, axis=1)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.predictions, labels), tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        # weight_decay = tf.reduce_mean(
            # tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = tf.reduce_mean(self.cross_entropy) #+ weight_decay
        # tf.summary.scalar('loss', self.loss)

        # self.global_step = tf.train.get_or_create_global_step()
        # self.learning_rate = tf.train.exponential_decay(initial_learning_rate,
        #                                                 self.global_step,
        #                                                 decay_steps,
        #                                                 decay_factor,
        #                                                 staircase=True)
        # tf.summary.scalar('lr', self.learning_rate)
        self.opt = tf.train.GradientDescentOptimizer(
            initial_learning_rate).minimize(self.loss)
        # self.merged = tf.summary.merge_all()

    def _conv2d(self, inputs, name, out_channels):
        # regularizer = tf.contrib.layers.l2_regularizer(5e-2)
        conv_relu = tf.layers.conv2d(
            inputs, filters=out_channels, kernel_size=[3, 3],
            kernel_initializer=tf.initializers.truncated_normal(stddev=5e-2),
            name=name)
        # tf.summary.histogram(name, conv_relu)
        return conv_relu

    def _max_pool(self, conv_relu, name):
        pool = tf.layers.max_pooling2d(
            conv_relu, pool_size=[2, 2], strides=2, name=name)
        # tf.summary.histogram(name, pool)
        return pool

    def _build_conv2d_block(self, inputs, name, out_channels, num_layers):
        conv_relu = inputs
        for i in range(num_layers):
            conv_relu = self._conv2d(conv_relu, '{}_{}'.format(
                name, i), out_channels)

        pool = self._max_pool(conv_relu, '{}_pool'.format(name))
        return pool

    def _fc(self, inputs, name, ouput_size, activate_func=tf.nn.relu):
        fc = tf.layers.dense(
            inputs, units=ouput_size,
            kernel_initializer=tf.initializers.truncated_normal(stddev=4e-2),
            activation=activate_func,
            name=name)
        # if activate_func is not None:
        #     fc = tf.contrib.layers.dropout(fc,
        #                                    keep_prob=0.5,
        #                                    is_training=self.is_training,
        #                                    scope=name + 'dropout')
        # tf.summary.histogram(name, fc)
        return fc
