import pandas as pd
import tensorflow as tf
from zipfile import ZipFile
import os


def read_labels_from_file(label_path):
    data = pd.read_csv(label_path)
    return data.id, data.breed


def parse_data(filename, label, new_size=(128, 128)):
    img_string = tf.read_file(filename)
    img_decoded = tf.image.decode_jpeg(img_string)
    image_resized = tf.image.resize_images(img_decoded, new_size)

    feature = np.array(img, dtype=np.float32) / 255.
    return img, label


def input_fn(label_path):
    filenames, labels = read_labels_from_file(label_path)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_data).batch(4)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    return next_element


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


if __name__ == '__main__':
    unzip_data('data/all.zip')
