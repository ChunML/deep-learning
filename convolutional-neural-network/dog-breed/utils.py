import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from zipfile import ZipFile
import os
import numpy as np
from PIL import Image
from collections import defaultdict
from time import time

data_root = './data/all'


def read_labels_from_file(data_root, label_path):
    data = pd.read_csv(label_path)
    data['fn'] = data.id.map(lambda x: os.path.join(
        data_root, 'train', x + '.jpg'))
    int_to_breed = data.breed.unique()
    breed_to_int = dict((v, k) for k, v in enumerate(int_to_breed))
    data['int_breed'] = data.breed.map(lambda x: breed_to_int[x])
    return data.fn, data.int_breed, int_to_breed, breed_to_int


def parse_data(filename, label=None, new_size=224):
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
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(parse_data).batch(1)

    iterator = dataset.make_one_shot_iterator()

    next_element = iterator.get_next()
    return next_element


def train_input_fn(filenames, labels, batch_size, num_train_files):
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
    folder = filepath[:filepath.find('.')]
    with ZipFile(filepath) as zipf:
        zipf.extractall(folder)

    filenames = [f for f in os.listdir(folder) if 'zip' in f]
    for filename in filenames:
        filename = os.path.join(folder, filename)
        sub_folder = filename[:filepath.find('.')]
        with ZipFile(filename) as zipf:
            zipf.extractall(sub_folder)


def converse_time(seconds):
    hours = seconds // 3600
    minutes = (seconds - hours * 3600) // 60
    seconds = seconds - hours * 3600 - minutes * 60
    return hours, minutes, seconds
