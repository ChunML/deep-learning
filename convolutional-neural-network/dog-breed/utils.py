import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from zipfile import ZipFile
import os
import numpy as np
from PIL import Image
from collections import defaultdict
from time import time
slim = tf.contrib.slim

data_root = './data/all'


def read_labels_from_file(data_root, label_path):
    ''' Get information from CSV files, including:
        1. Get image filenames
        2. Create dict to categorize labels
    Args:
        data_root: root path of data folder
        label_path: path to the CSV file

    Returns:
        fn: the fool paths of all training images
        int_breed: the training labels in integer
        int_to_breed: dict to convert indices to labels
        breed_to_int: dict to convert labels to indices
    '''
    data = pd.read_csv(label_path)
    data['fn'] = data.id.map(lambda x: os.path.join(
        data_root, 'train', x + '.jpg'))
    int_to_breed = data.breed.unique()
    breed_to_int = dict((v, k) for k, v in enumerate(int_to_breed))
    data['int_breed'] = data.breed.map(lambda x: breed_to_int[x])
    return data.fn, data.int_breed, int_to_breed, breed_to_int


def parse_data(filename, label=None, new_size=224):
    ''' Processing images and labels
        1. Read image
        2. Resize image to size [new_size, new_size]
        3. Subtract by mean from ImageNet
        4. Cast int labels to tf.int64
    Args:
        filename: full paths to images
        label: labels in interger (None if in eval mode)
        new_size: new size to resize image

    Returns:
        tuple of processed images and labels
    '''
    means = [123.68, 116.779, 103.939]
    img_string = tf.read_file(filename)
    img = tf.image.decode_jpeg(img_string)
    img = tf.image.resize_images(
        img, (new_size, new_size))
    img.set_shape([224, 224, 3])
    img = tf.to_float(img)
    channels = tf.split(axis=2, num_or_size_splits=3, value=img)
    for i in range(3):
        channels[i] -= means[i]
    if label == None:
        return tf.concat(axis=2, values=channels)

    label = tf.cast(label, tf.int64)

    return tf.concat(axis=2, values=channels), label


def eval_input_fn(filenames):
    ''' Create input_fn for eval mode
    Args:
        filenames: full paths to images

    Returns:
        op to iterate through test images
    '''
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(parse_data).batch(1)

    iterator = dataset.make_one_shot_iterator()

    next_element = iterator.get_next()
    return next_element


def train_input_fn(filenames, labels, batch_size, num_train_files):
    ''' Create input_fn for training mode:
        1. Split data into training data and validation data
        2. Create seperate init_fn for training data and validation data
    Args:
        filenames: full paths to images
        labels: labels in interger
        batch_size: batch size
        num_train_files: number of training files

    Returns:
        op to iterate through data
        init_op: op for training
        val_init_op: op for validation
    '''
    train_filenames = filenames[:num_train_files]
    train_labels = labels[:num_train_files]
    val_filenames = filenames[num_train_files:]
    val_labels = labels[num_train_files:]
    dataset = tf.data.Dataset.from_tensor_slices(
        (train_filenames, train_labels))
    dataset = dataset.map(parse_data).shuffle(
        len(train_filenames)).repeat().batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_filenames, val_labels))
    val_dataset = val_dataset.map(parse_data).batch(batch_size)

    iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                               dataset.output_shapes)
    next_element = iterator.get_next()

    init_op = iterator.make_initializer(dataset)
    val_init_op = iterator.make_initializer(val_dataset)

    return next_element, init_op, val_init_op


def unzip_data(filepath):
    ''' Unzip downloaded data
    Args:
        filepath: path to zip file

    Returns:
        Nothing
    '''
    folder = filepath[:filepath.find('.')]
    with ZipFile(filepath) as zipf:
        zipf.extractall(folder)

    filenames = [f for f in os.listdir(folder) if 'zip' in f]
    for filename in filenames:
        filename = os.path.join(folder, filename)
        sub_folder = filename[:filepath.find('.')]
        with ZipFile(filename) as zipf:
            zipf.extractall(sub_folder)


def get_variables_to_restore_and_initializer(exclude_vars, use_slim=False):
    ''' Restore weights from pretrained ImageNet weights
        1. Create list of variables to restore exclude ones from exclude_vars
           If not use_slim, we need to create a dict to map new variable names
        2. Create list of variables to initialize
           This will be the ones in exclude_vars
    Args:
        exclude_vars: list of regexes of variables to ignore when restoring
                      ex: 'fc8' -> ignore 'fc8/weights', 'fc8/biases'
        use_slim: whether using slim model or not

    Returns:
        variables_to_restore: list of variables to be restored
        variables_to_initialize: list of variables to be initialized
    '''
    trainable_variables = tf.trainable_variables()

    if use_slim:
        variables_to_restore = [v for v in trainable_variables if 'fc8' not in v.op.name ]
        variables_to_initialize = [v for v in trainable_variables if 'fc8' in v.op.name]
    else:
        variables_to_restore = {}
        variables_to_initialize = []
        for v in trainable_variables:
            v_name = v.op.name
            kernel_or_bias = v_name[v_name.rfind('/') + 1:]
            if kernel_or_bias == 'kernel':
                key = v_name[:v_name.rfind('/') + 1] + 'weights'
            else:
                key = v_name[:v_name.rfind('/') + 1] + 'biases'
            if 'fc8' in v_name:
                variables_to_initialize.append(v)
            else:
                variables_to_restore[key] = v
    return variables_to_restore, variables_to_initialize


def converse_time(seconds):
    hours = seconds // 3600
    minutes = (seconds - hours * 3600) // 60
    seconds = seconds - hours * 3600 - minutes * 60
    return hours, minutes, seconds
