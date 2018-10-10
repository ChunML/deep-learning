import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from read_data import read_data

data_path = './hour.csv'
iterations = 2000
learning_rate = 0.8
hidden_size = 16
output_size = 1


class NeuralNetwork(object):
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        self.weights_input_to_hidden = np.random.normal(0.0, self.input_size ** -0.5,
                                                        (self.input_size, self.hidden_size))
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_size ** -0.5,
                                                         (self.hidden_size, self.output_size))

        self.activation_function = lambda x: 1.0 / (1 + np.exp(-x))

    def forward(self, X):
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        outputs = np.dot(hidden_outputs, self.weights_hidden_to_output)

        return outputs, hidden_outputs

    def backprop(self, outputs, hidden_outputs, y, X, delta_weights_h_o, delta_weights_i_h):
        error = y - outputs

        hidden_error = np.dot(error, self.weights_hidden_to_output.T)
        output_error_terms = error
        hidden_error_terms = hidden_error * \
            hidden_outputs * (1 - hidden_outputs)

        delta_weights_h_o += np.dot(hidden_outputs.T, output_error_terms)
        delta_weights_i_h += np.dot(X.T, hidden_error_terms)

        return delta_weights_h_o, delta_weights_i_h

    def update_weights(self, delta_weights_h_o, delta_weights_i_h, n_record):
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_record
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_record

    def train(self, features, targets):
        n_record = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):
            X = X[None, :]
            #y = y[:, None]
            outputs, hidden_outputs = self.forward(X)
            delta_weights_h_o, delta_weights_i_h = self.backprop(outputs, hidden_outputs,
                                                                 y, X,
                                                                 delta_weights_h_o,
                                                                 delta_weights_i_h)
        self.update_weights(delta_weights_h_o, delta_weights_i_h, n_record)


def main(features, targets, test_features, test_targets, scaled_features):
    def mse_loss(y_, y):
        return np.mean((y_ - y) ** 2)

    input_size = features.shape[1]
    network = NeuralNetwork(input_size, hidden_size,
                            output_size, learning_rate)

    losses = {'train': [], 'validation': []}

    for ii in range(iterations):
        batch = np.random.choice(features.index, size=128)
        X, y = features.iloc[batch].values, targets.iloc[batch]['cnt'][:, None]

        network.train(X, y)

        train_loss = mse_loss(network.forward(features)[
                              0].T, targets['cnt'].values)
        val_loss = mse_loss(network.forward(test_features)[
                            0].T, test_targets['cnt'].values)

        sys.stdout.write('\nProgress: {:2.1f}%'.format(100 * ii / float(iterations)) +
                         ' Training loss: {:2.4f}'.format(train_loss) +
                         ' Val loss: {:2.4f}'.format(val_loss))

        losses['train'].append(train_loss)
        losses['validation'].append(val_loss)

    mean, std = scaled_features['cnt']
    predictions = network.forward(test_features)[0].T * std + mean

    return losses, predictions


if __name__ == '__main__':
    features, targets, test_features, test_targets, scaled_features, dteday = read_data(
        data_path)
    losses, predictions = main(
        features, targets, test_features, test_targets, scaled_features)
    plt.plot(losses['train'], label='Training Loss')
    plt.plot(losses['validation'], label='Validation Loss')
    plt.legend()
    plt.ylim((0.0, 0.8))
    plt.show()

    mean, std = scaled_features['cnt']
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(predictions[0], label='Prediction')
    ax.plot((test_targets['cnt'] * std + mean).values, label='Data')
    # ax.set_xlim(right=len(predictions))
    ax.legend()

    dates = pd.to_datetime(dteday.iloc[test_targets.index])
    dates = dates.apply(lambda d: d.strftime('%b %d'))
    ax.set_xticks(np.arange(len(dates))[12::24])
    _ = ax.set_xticklabels(dates[12::24], rotation=45)
    plt.show()
