import matplotlib.pyplot as plt
from urllib.request import urlretrieve
import os
from zipfile import ZipFile
from tqdm import tqdm
import numpy as np
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split

train_file_url = 'https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip'
test_file_url = 'https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip'
train_filename = 'notMNIST_train.zip'
test_filename = 'notMNIST_test.zip'
int_to_label = ['A', 'B', 'C', 'D', 'E', 'F',
                'G', 'H', 'I', 'J', 'K', 'L',
                'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
label_to_int = dict((v, k) for (k, v) in enumerate(int_to_label))


def download_data(url, filename):
    ''' Download data from URL.
    Args:
        url: URL to get file from
        filename: filename to save as

    Returns:
        nothing
    '''
    if not os.path.isfile(filename):
        print('[INFO] Downloading {}...'.format(filename))
        urlretrieve(url, filename)
        print('[INFO] {} has been downloaded!'.format(filename))
    else:
        print('[INFO] {} has already existed. Download skipped!'.format(filename))


def uncompress_features_labels(filename):
    ''' Uncompress downloaded zip file to features and labels.
    Args:
        filename: filename to uncompress

    Returns:
        features: features with shape [num_of_files, feature_size]
        labels: labels with shape [num_of_files, label_size]
    '''
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


def normalize_grayscale(image_data, a=-1, b=1):
    ''' Normalize pixel value to range [a, b].
    Args:
        image_data: image data with shape [num_of_files, feature_size]
        a: lower band
        b: upper band

    Returns:
        image_data normalized to [a, b]
    '''
    xmin = 0
    xmax = 255
    print('[INFO] Image normalized!')
    return 0.1 + (image_data - xmin) * (b - a) / (xmax - xmin)


def convert_labels(labels):
    ''' Binarize label from string values to sparse arrays.
    Args:
        labels: label data as string values

    Returns:
        label data as sparse arrays of float32
    '''
    labels = [label_to_int[l] for l in labels]
    print('[INFO] Label binarized!')
    return np.array(labels, dtype=np.int32)


def dump_to_pickle(data, filename):
    ''' Save processed data to pickle file for future use.
    Args:
        data: processed data as a Dict
        filename: pickle filename to dump

    Returns:
        nothing
    '''
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print('[INFO] Data dumped to pickle file at {}.'.format(filename))


def process_data(filename):
    ''' Process data for training and testing,
    including all necessary steps above.

    Args:
        data: processed data as a Dict
        filename: pickle filename to dump

    Returns:
        nothing
    '''
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print('[INFO] Data successfully loaded!')
        return (data['train_features'], data['train_labels']), (data['valid_features'], data['valid_labels']), (data['test_features'], data['test_labels'])

    else:
        download_data(train_file_url, train_filename)
        download_data(test_file_url, test_filename)

        train_features, train_labels = uncompress_features_labels(
            train_filename)
        test_features, test_labels = uncompress_features_labels(test_filename)

        train_features = normalize_grayscale(train_features)
        test_features = normalize_grayscale(test_features)

        train_labels = convert_labels(train_labels)
        test_labels = convert_labels(test_labels)

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


def plot_result_sample(features, predictions):
    ''' Plot randomly 16 test images with their predicted labels

    Args:
        features: features of test data
        predictions: predictions from network

    Returns:
        nothing
    '''
    random_indices = np.random.permutation(len(features))
    features = features[random_indices]
    predictions = predictions[random_indices]

    _, axes = plt.subplots(ncols=4, nrows=4)
    for i in range(16):
        feature = (features[i] * 255).reshape((28, 28)).astype(np.int32)
        img = Image.fromarray(feature)
        axes[int(i / 4), i % 4].imshow(img)
        axes[int(i / 4), i % 4].axis('off')
        axes[int(i / 4), i % 4].text(2, 2,
                                     int_to_label[predictions[i]],
                                     color='red',
                                     fontsize=20)
    plt.show()
