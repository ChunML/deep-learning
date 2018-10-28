import tensorflow as tf
import numpy as np
from functools import reduce


def conv2d(inputs, filters, name, padding='same', kernel_size=[3, 3], activation=tf.nn.relu):
    net = tf.layers.conv2d(inputs, filters=filters,
                           kernel_size=kernel_size,
                           padding=padding,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                           bias_initializer=tf.zeros_initializer(),
                           name=name)
    return net

def conv2d_block(conv, num_layers, filters, name):
    with tf.variable_scope(name):
        for i in range(num_layers):
            conv = conv2d(conv, filters=filters, name=name + '_{}'.format(i+1))
    return conv

def vgg16(images, num_classes, is_training):
    with tf.variable_scope('vgg_16'):
        net = conv2d_block(images, 2, 64, 'conv1')
        net = tf.layers.max_pooling2d(net, 2, 2, padding='valid', name='pool1')
        net = conv2d_block(net, 2, 128, 'conv2')
        net = tf.layers.max_pooling2d(net, 2, 2, padding='valid', name='pool2')
        net = conv2d_block(net, 3, 256, 'conv3')
        net = tf.layers.max_pooling2d(net, 2, 2, padding='valid', name='pool3')
        net = conv2d_block(net, 3, 512, 'conv4')
        net = tf.layers.max_pooling2d(net, 2, 2, padding='valid', name='pool4')
        net = conv2d_block(net, 3, 512, 'conv5')
        net = tf.layers.max_pooling2d(net, 2, 2, padding='valid', name='pool5')

        net = conv2d(net, 4096, 'fc6', kernel_size=[7, 7], padding='valid')
        net = tf.layers.dropout(net, training=is_training, name='dropout6')
        net = conv2d(net, 4096, 'fc7', kernel_size=[1, 1])
        net = tf.layers.dropout(net, training=is_training, name='dropout7')

        net = conv2d(net, num_classes, 'fc8', kernel_size=[1, 1])
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
    return net
