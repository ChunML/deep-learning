import tensorflow as tf
import numpy as np
from functools import reduce
from time import time
from utils import converse_time
from collections import defaultdict


class Network(object):
    def __init__(self, images, labels, ouput_size, initial_learning_rate, decay_steps, decay_factor=0.94):
        with tf.name_scope('conv1') as scope:
            pool1 = self._build_conv2d_block(images, scope, 64, 2)
        with tf.name_scope('conv2') as scope:
            pool2 = self._build_conv2d_block(pool1, scope, 128, 2)
        with tf.name_scope('conv3') as scope:
            pool3 = self._build_conv2d_block(pool2, scope, 256, 3)
        with tf.name_scope('conv4') as scope:
            pool4 = self._build_conv2d_block(pool3, scope, 512, 3)
        with tf.name_scope('conv5') as scope:
            pool5 = self._build_conv2d_block(pool4, scope, 512, 3)

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
        tf.summary.scalar('accuracy', self.accuracy)

        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        self.loss = tf.reduce_mean(self.cross_entropy)
        tf.summary.scalar('loss', self.loss)

        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                                        self.global_step,
                                                        decay_steps,
                                                        decay_factor,
                                                        staircase=True)
        self.opt = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(self.loss)
        self.merged = tf.summary.merge_all()

    def _conv2d(self, inputs, name, out_channels):
        conv_relu = tf.contrib.layers.conv2d(
            inputs, out_channels, [3, 3], scope=name)
        tf.summary.histogram(name, conv_relu)
        return conv_relu

    def _max_pool(self, conv_relu, name):
        pool = tf.contrib.layers.max_pool2d(conv_relu, [2, 2], scope=name)
        tf.summary.histogram(name, pool)
        return pool

    def _build_conv2d_block(self, inputs, name, out_channels, num_layers):
        conv_relu = inputs
        for i in range(num_layers):
            conv_relu = self._conv2d(conv_relu, '{}_{}'.format(
                name, i), out_channels)

        pool = self._max_pool(conv_relu, '{}_pool'.format(name))
        return pool

    def _fc(self, inputs, name, ouput_size, activate_func=tf.nn.relu):
        fc = tf.contrib.layers.fully_connected(
            inputs, ouput_size, activation_fn=activate_func)
        tf.summary.histogram(name, fc)
        return fc

    def train(self, num_epochs, num_batches, checkpoint_fn='checkpoints/dogs.ckpt'):
        train_info = defaultdict(list)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('logdir', sess.graph)
            sess.run(tf.global_variables_initializer())
            for epoch_i in range(num_epochs):
                for batch_i in range(num_batches):
                    start_time = time()
                    _, loss, acc = sess.run(
                        [self.opt, self.loss, self.accuracy])
                    if batch_i % 50 == 0:
                        summary = sess.run(self.merged)
                        train_writer.add_summary(summary, self.global_step)
                    train_info['loss'].append(loss)
                    train_info['acc'].append(acc)
                    batch_time = time() - start_time
                    batches_left = (num_epochs - epoch_i - 1) * \
                        num_batches + num_batches - batch_i
                    time_left = batch_time * batches_left
                    hours, minutes, seconds = converse_time(time_left)
                    print('Epoch: {} Batch: {}/{} Loss: {:.4f} Accuracy: {:.4f} Time left: {:.0f}:{:.0f}:{:.0f}'.format(
                        epoch_i + 1, batch_i, num_batches, loss, acc, hours, minutes, seconds))
            saver.save(sess, checkpoint_fn)
        return train_info
