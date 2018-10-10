import tensorflow as tf
from read_data import read_data
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt

data_path = './hour.csv'
hidden_size = 16
output_size = 1
batch_size = 128
learning_rate = 0.45
iterations = 2000

class TFNeuralNetwork(object):
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.inputs_ = tf.placeholder(tf.float32, [None, input_size])
        self.outputs_ = tf.placeholder(tf.float32, [None, output_size])
        self.lr = lr

        with tf.variable_scope('tf_net'):
            with tf.variable_scope('hidden'):
                W = tf.get_variable(
                    'weights', [input_size, hidden_size],
                    initializer=tf.random_normal_initializer(stddev=input_size ** -0.5))
                self.hidden = tf.nn.sigmoid(tf.matmul(self.inputs_, W))

            with tf.variable_scope('predictions'):
                W = tf.get_variable(
                    'weights', [hidden_size, output_size],
                    initializer=tf.random_normal_initializer(stddev=input_size ** -0.5))
                self.predictions = tf.matmul(self.hidden, W)

            #self.loss = tf.losses.mean_squared_error(self.outputs_, self.predictions)
            self.loss = tf.reduce_mean(tf.square(self.outputs_- self.predictions))
            self.opt = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)

def main(features, targets, test_features, test_targets):
    input_size = features.shape[1]
    network = TFNeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    losses = {'train': [], 'validation': []}

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for ii in range(iterations):
            batch = np.random.choice(features.shape[0], size=batch_size)
            _ = sess.run(network.opt,
                         feed_dict={network.inputs_: features.iloc[batch].values,
                                    network.outputs_: targets[batch]})

            train_loss = network.loss.eval(
                feed_dict={network.inputs_: features.values,
                           network.outputs_: targets})
            val_loss = network.loss.eval(
                feed_dict={network.inputs_: test_features.values,
                           network.outputs_: test_targets})
            sys.stdout.write('\nProgress: {:2.1f}%'.format(100 * ii / float(iterations)) + \
                             ' Training loss: {:2.4f}'.format(train_loss) + \
                             ' Val loss: {:2.4f}'.format(val_loss))
            losses['train'].append(train_loss)
            losses['validation'].append(val_loss)
        predictions = network.predictions.eval(
            feed_dict={network.inputs_: test_features.values,
                       network.outputs_: test_targets})
    return losses, predictions

if __name__ == '__main__':
    features, targets, test_features, test_targets, scaled_features, dteday = read_data(data_path)
    targets = targets['cnt'][:, None]
    test_targets = test_targets['cnt'][:, None]
    losses, predictions = main(features, targets, test_features, test_targets)

    plt.plot(losses['train'], label='Training Loss')
    plt.plot(losses['validation'], label='Validation Loss')
    plt.legend()
    plt.ylim((0.0, 1.0))
    plt.show()

    mean, std = scaled_features['cnt']
    predictions = predictions[:, 0].T * std + mean
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(predictions, label='Prediction')
    ax.plot(test_targets[:, 0] * std + mean, label='Data')
    #ax.set_xlim(right=len(predictions))
    ax.legend()
    dates = pd.to_datetime(dteday.iloc[test_features.index])
    dates = dates.apply(lambda d: d.strftime('%b %d'))
    ax.set_xticks(np.arange(len(dates))[12::24])
    _ = ax.set_xticklabels(dates[12::24], rotation=45)
    plt.show()
