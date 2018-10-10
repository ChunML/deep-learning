from urllib.request import urlretrieve
import os
from zipfile import ZipFile
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
import pickle
from sklearn.model_selection import train_test_split

train_file_url = 'https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip'
test_file_url = 'https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip'
train_filename = 'notMNIST_train.zip'
test_filename = 'notMNIST_test.zip'

def download_data(url, filename):
    if not os.path.isfile(filename):
        urlretrieve(url, filename)
        print('[INFO] Zip files has been downloaded!')
    else:
        print('[INFO] Zip files has already existed!')

def uncompress_features_labels(filename):
    features = []
    labels = []

    with ZipFile(filename) as zipf:
        filenames_pbar = tqdm(zipf.namelist(), unit='files')

        for filename in filenames_pbar:
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()

                    feature = np.array(image, dtype=np.float32).flatten()
                
                label = os.path.split(filename)[1][0]

                features.append(feature)
                labels.append(label)
    print('[INFO] Features and labels uncompressed!')
    return np.array(features), np.array(labels)

def normalize_grayscale(image_data):
    xmin = 0
    xmax = 255
    a = 0.1
    b = 0.9
    print('[INFO] Image normalized!')
    return 0.1 + (image_data - xmin) * (b - a) / (xmax - xmin)

def binarize_label(labels):
    encoder = LabelBinarizer()
    encoder.fit(labels)
    labels = encoder.transform(labels)
    print('[INFO] Label binarized!')
    return labels.astype(np.float32)

def dump_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print('[INFO] Data dumped to pickle file at {}.'.format(filename))

def process_data(filename):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print('[INFO] Data successfully loaded!')
        return (data['train_features'], data['train_labels']), (data['valid_features'], data['valid_labels']), (data['test_features'], data['test_labels'])

    else:
        download_data(train_file_url, train_filename)
        download_data(test_file_url, test_filename)

        train_features, train_labels = uncompress_features_labels(train_filename)
        test_features, test_labels = uncompress_features_labels(test_filename)

        train_features = normalize_grayscale(train_features)
        test_features = normalize_grayscale(test_features)

        train_labels = binarize_label(train_labels)
        test_labels = binarize_label(test_labels)

        train_features, valid_features, train_labels, valid_labels = train_test_split(
            train_features, train_labels,
            test_size=0.1)

        data = {
            'train_features': train_features,
            'train_labels': train_labels,
            'valid_features': valid_features,
            'valid_labels': valid_labels,
            'test_features': test_features,
            'test_labels': test_labels
        }
        dump_to_pickle(data, filename)
        print('[INFO] Data successfully loaded!')
        return (train_features, train_labels), (valid_features, valid_labels), (test_features, test_labels)
