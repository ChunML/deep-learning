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


def read_labels_from_file(label_path):
    data = pd.read_csv(label_path)
    data['fn'] = data.id.map(lambda x: os.path.join(
        data_root, 'train', x + '.jpg'))
    int_to_breed = data.breed.unique()
    breed_to_int = dict((v, k) for k, v in enumerate(int_to_breed))
    data['int_breed'] = data.breed.map(lambda x: breed_to_int[x])
    return data.fn, data.int_breed, int_to_breed, breed_to_int


def parse_data(filename, label, new_size=224):
    img_string = tf.read_file(filename)
    img_decoded = tf.image.decode_jpeg(img_string)
    image_resized = tf.image.resize_images(
        img_decoded, (new_size, new_size)) - [123.68, 116.779, 103.939]
    image_resized.set_shape([224, 224, 3])
    label = tf.cast(label, tf.int64)

    return image_resized, label


def input_fn(filenames, labels, batch_size, num_train_files):
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
