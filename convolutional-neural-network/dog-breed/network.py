import tensorflow as tf
import numpy as np
from functools import reduce
from time import time
from utils import converse_time
from collections import defaultdict


class Network(object):
    def __init__(self, images, labels, image_size, channel, ouput_size):
        with tf.name_scope('conv1') as scope:
            pool1 = self._build_conv2d_block(images, scope, 3, 64, 2)
        with tf.name_scope('conv2') as scope:
            pool2 = self._build_conv2d_block(pool1, scope, 64, 128, 2)
        with tf.name_scope('conv3') as scope:
            pool3 = self._build_conv2d_block(pool2, scope, 128, 128, 3)
        with tf.name_scope('conv4') as scope:
            pool4 = self._build_conv2d_block(pool3, scope, 128, 256, 3)
        with tf.name_scope('conv5') as scope:
            pool5 = self._build_conv2d_block(pool4, scope, 256, 512, 3)

        pool5_shape = pool5.get_shape().as_list()
        flatten = tf.reshape(
            pool5, [-1, reduce(lambda x, y: x * y, pool5_shape[1:])])

        fc6 = self._fc(flatten, 'fc6', 2048)
        fc6_dropout = tf.nn.dropout(fc6, keep_prob=0.5)
        fc7 = self._fc(fc6_dropout, 'fc7', 1024)
        fc6_dropout = tf.nn.dropout(fc7, keep_prob=0.5)
        logits = self._fc(
            fc6_dropout, 'logits', ouput_size, activate_func=None)
        self.predictions = tf.argmax(logits, axis=1)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.predictions, labels), tf.float32))

        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        self.loss = tf.reduce_mean(self.cross_entropy)
        self.opt = tf.train.GradientDescentOptimizer(0.001).minimize(self.loss)

    def train(self, num_epochs, num_batches):
        train_info = defaultdict(list)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch_i in range(num_epochs):
                for batch_i in range(num_batches):
                    start_time = time()
                    _, loss, acc = sess.run(
                        [self.opt, self.loss, self.accuracy])
                    train_info['loss'].append(loss)
                    train_info['acc'].append(acc)
                    batch_time = time() - start_time
                    batches_left = (num_epochs - epoch_i - 1) * \
                        num_batches + num_batches - batch_i
                    time_left = batch_time * batches_left
                    hours, minutes, seconds = converse_time(time_left)
                    print('Epoch: {} Batch: {}/{} Loss: {:.4f} Accuracy: {:.4f} Time left: {:.0f}:{:.0f}:{:.0f}'.format(
                        epoch_i + 1, batch_i, num_batches, loss, acc, hours, minutes, seconds))

        return train_info

    def _conv2d(self, inputs, name, in_channels, out_channels):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable(
                'weights',
                [3, 3, in_channels, out_channels],
                initializer=tf.truncated_normal_initializer(stddev=5e-2),
                dtype=tf.float32)
            biases = tf.get_variable(
                'biases',
                [out_channels],
                initializer=tf.constant_initializer(0.0),
                dtype=tf.float32)
            conv = tf.nn.conv2d(inputs, weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')
            conv_relu = tf.nn.relu(tf.nn.bias_add(
                conv, biases), name=scope.name)
        return conv_relu

    def _max_pool(self, conv_relu, name):
        with tf.variable_scope(name) as scope:
            return tf.nn.max_pool(conv_relu, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name=scope.name)

    def _build_conv2d_block(self, inputs, name, in_channels, out_channels, num_layers):
        for i in range(num_layers):
            conv_relu = self._conv2d(inputs, '{}_{}'.format(
                name, i), in_channels, out_channels)
            inputs = conv_relu
            in_channels = out_channels

        pool = self._max_pool(conv_relu, '{}_pool'.format(name))
        return pool

    def _fc(self, inputs, name, ouput_size, activate_func=tf.nn.relu):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable(
                'weights', [inputs.get_shape()[-1], ouput_size],
                initializer=tf.truncated_normal_initializer(stddev=0.04),
                dtype=tf.float32)
            biases = tf.get_variable(
                'biases', [ouput_size],
                initializer=tf.constant_initializer(0.0),
                dtype=tf.float32)
            if activate_func:
                fc = activate_func(
                    tf.matmul(inputs, weights) + biases, name=scope.name)
            else:
                fc = tf.identity(tf.matmul(inputs, weights) +
                                 biases, name=scope.name)
        return fc
