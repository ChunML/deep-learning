from keras import Sequential
from keras.layers import Dense
import keras.models as models
from keras.callbacks import Callback
from keras.initializers import TruncatedNormal, Zeros
from keras.optimizers import SGD
import numpy as np
import os


class Network(object):
    def __init__(self, input_size, hidden_size, output_size, lr):
        ''' Train and validate network using notMNIST data.
        Args:
            input_size: size of input feature
            hidden_size: size of hidden layer
            output_size: size of output layer
            lr: learning rate
            name: network's scope name

        Returns:
            nothing
        '''
        self.model = Sequential()
        self.model.add(Dense(hidden_size,
                             kernel_initializer=TruncatedNormal(
                                 stddev=input_size ** -0.5),
                             bias_initializer=Zeros(),
                             activation='relu',
                             input_shape=(input_size,)))
        self.model.add(Dense(output_size,
                             kernel_initializer=TruncatedNormal(
                                 stddev=hidden_size ** -0.5),
                             bias_initializer=Zeros(),
                             activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=SGD(lr=lr), metrics=['accuracy'])

    def train(self, num_epochs, batch_size,
              train_features, train_labels,
              valid_features, valid_labels,
              checkpoint_fn):
        ''' Train and validate network using notMNIST data.
        Args:
            num_epochs: number of epochs to train
            batch_size: batch_size
            train_features: training features with shape [batch_size, feature_size]
            train_labels: training labels with shape [batch_size, label_size]
            valid_features: validation features with shape [batch_size, feature_size]
            valid_labels: validation labels with shape [batch_size, label_size]
            checkpoint_fn: checkpoint filename to save after training

        Returns:
            train_losses: training losses
            val_losses: validation losses
        '''

        num_of_batches = len(train_features) // batch_size

        history = self.model.fit(x=train_features, y=train_labels,
                                 validation_data=(
                                     valid_features, valid_labels),
                                 epochs=num_epochs,
                                 batch_size=batch_size)
        self.model.save(checkpoint_fn)
        print('[INFO] Training weights have been '
              'successfully saved to {}.'.format(checkpoint_fn))
        return history.history['loss'], history.history['val_loss']


    def inference(self, test_features, checkpoint_fn):
        ''' Test network using notMNIST test data.
        Args:
            test_features: test features with shape [batch_size, feature_size]
            checkpoint_fn: trained checkpoint filename

        Returns:
            predictions: predictions made by the network
        '''
        model = models.load_model(checkpoint_fn)
        print('[INFO] Training weights have been '
              'successfully loaded from {}.'.format(checkpoint_fn))
        predictions = model.predict(test_features).argmax(axis=1)
        return predictions
