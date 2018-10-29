import tensorflow as tf
import numpy as np
from functools import reduce


def conv2d(inputs, filters, name, padding='same', kernel_size=[3, 3], activation=tf.nn.relu):
    ''' Create a conv2d layer using tf.layers
    Args:
        inputs: the input tensor of con2d layer
        filters: number of output channels
        name: scope name
        padding: either same or valid
        kernel_size: the kernel size of this conv2d layer
        activation: activation function

    Returns:
        conv: output of con2d layer
    '''
    conv = tf.layers.conv2d(inputs, filters=filters,
                           kernel_size=kernel_size,
                           padding=padding,
                           activation=activation,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                           name=name)
    return conv

def conv2d_block(conv, num_layers, filters, name):
    ''' Create a block of conv2d layers
    Args:
        conv: the input tensor of this block
        num_layers: number of conv2d layers
        filters: number of output channels
        name: scope name

    Returns:
        conv: output of this block
    '''
    with tf.variable_scope(name):
        for i in range(num_layers):
            conv = conv2d(conv, filters=filters, name=name + '_{}'.format(i+1))
    return conv

def vgg16(images, num_classes, is_training):
    ''' Build the customized version of VGG16
    Args:
        images: the input batch of images
                of shape [batch_size, 224, 224, 3]
        num_classes: number of classes to specify
                     the last layer's output channel
        is_training: whether in training mode or not,
                     needed for dropout layers

    Returns:
        net: output of the last layer (fc8)
    '''
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

        net = conv2d(net, num_classes, 'fc8',
                     kernel_size=[1, 1], activation=None)
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
    return net
