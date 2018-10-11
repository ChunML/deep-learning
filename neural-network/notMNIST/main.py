from utils import process_data, plot_result_sample, int_to_label
import tensorflow as tf
from network import Network
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

filename = 'notMNIST.pkl'


def train(net, num_epochs, batch_size,
          train_features, train_labels,
          valid_features, valid_labels,
          checkpoint_fn):
    ''' Train and validate network using notMNIST data.
    Args:
        net: Network defined in ./network.py
        num_epochs: number of epochs to train
        batch_size: batch_size
        train_features: training features with shape [batch_size, feature_size]
        train_labels: training labels with shape [batch_size, label_size]
        valid_features: validation features with shape [batch_size, feature_size]
        valid_labels: validation labels with shape [batch_size, label_size]
        checkpoint_fn: checkpoint filename to save after training

    Returns:
        nothing
    '''
    train_losses, val_losses = net.train(num_epochs, batch_size,
                                         train_features, train_labels,
                                         valid_features, valid_labels,
                                         checkpoint_fn)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.show()


def test(net, test_features, checkpoint_fn):
    ''' Test network using notMNIST test data.
    Args:
        net: Network defined in ./network.py
        test_features: test features with shape [batch_size, feature_size]
        checkpoint_fn: trained checkpoint filename

    Returns:
        nothing
    '''
    try:
        predictions = net.inference(test_features, checkpoint_fn)
        predictions = np.argmax(predictions, axis=1)
    except Exception:
        print('[ERROR] Checkpoint file does not exist. Train the network first!')

    accuracy = np.equal(predictions, test_labels,
                        dtype=np.int64).mean()
    print('[INFO] Accuracy on test set: {:.4f}.'.format(accuracy))
    print('[INFO] Plotting result samples...')
    plot_result_sample(test_features, predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--image-dir', default=None)
    parser.add_argument('--hidden-size', default=256)
    parser.add_argument('--batch-size', default=128)
    parser.add_argument('--checkpoint_fn', default='checkpoints/notMNIST.ckpt')
    parser.add_argument('--num-epochs', default=5)
    parser.add_argument('--lr', default=0.2)
    args = parser.parse_args()

    (train_features, train_labels),\
        (valid_features, valid_labels),\
        (test_features, test_labels) = process_data(filename)

    features_size = train_features.shape[1]
    labels_size = len(int_to_label)

    net = Network(input_size=features_size,
                  hidden_size=args.hidden_size,
                  output_size=labels_size,
                  lr=args.lr)

    if args.mode.lower() == 'train':
        train(net, args.num_epochs, args.batch_size,
              train_features, train_labels,
              valid_features, valid_labels,
              args.checkpoint_fn)
    elif args.mode.lower() == 'test':
        test(net, test_features, args.checkpoint_fn)
    else:
        print('[ERROR] Mode must be either "train" or "test"!')
