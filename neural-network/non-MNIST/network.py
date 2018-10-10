import tensorflow as tf
import numpy as np
import os


class Network(object):
    def __init__(self, input_size, hidden_size, output_size, lr, name='network'):
        with tf.name_scope(name):
            self.inputs = tf.placeholder(
                tf.float32, [None, input_size], name='features')
            self.labels = tf.placeholder(tf.float32, [None, output_size])
            with tf.variable_scope('hidden'):
                weights = tf.get_variable('weights', [input_size, hidden_size],
                                          initializer=tf.truncated_normal_initializer(
                                              stddev=input_size ** -0.5))
                biases = tf.get_variable('biases', [hidden_size],
                                         initializer=tf.zeros_initializer)
                hidden = tf.nn.relu(tf.matmul(self.inputs, weights) + biases)

            with tf.variable_scope('outputs'):
                weights = tf.get_variable('weights', [hidden_size, output_size],
                                          initializer=tf.truncated_normal_initializer(
                                              stddev=hidden_size ** -0.5))
                biases = tf.get_variable('biases', [output_size],
                                         initializer=tf.zeros_initializer)
                logits = tf.matmul(hidden, weights) + biases
            self.predictions = tf.argmax(logits, axis=1)
            correct_preds = tf.equal(tf.argmax(self.labels, axis=1),
                                     self.predictions)
            self.acc = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.labels, logits=logits)
            self.loss = tf.reduce_mean(cross_entropy)
            self.opt = tf.train.GradientDescentOptimizer(
                lr).minimize(self.loss)

    def train(self, num_epochs, batch_size,
              train_features, train_labels,
              valid_features, valid_labels,
              checkpoint_fn):
        saver = tf.train.Saver()

        num_of_batches = len(train_features) // batch_size
        train_losses = []
        val_losses = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(num_epochs):
                new_indices = np.random.permutation(
                    np.arange(len(train_features)))

                for batch_i in range(num_of_batches):
                    batch_indices = new_indices[batch_i *
                                                batch_size:(batch_i + 1) * batch_size]
                    loss, _ = sess.run([self.loss, self.opt],
                                       feed_dict={self.inputs: train_features[batch_indices],
                                                  self.labels: train_labels[batch_indices]})
                    if batch_i % 10 == 0:
                        val_loss = self.loss.eval(
                            feed_dict={self.inputs: valid_features,
                                       self.labels: valid_labels})
                        train_losses.append(loss)
                        val_losses.append(val_loss)
                    if batch_i % 100 == 0:
                        train_acc = self.acc.eval(
                            feed_dict={self.inputs: train_features,
                                       self.labels: train_labels})
                        val_acc = self.acc.eval(
                            feed_dict={self.inputs: valid_features,
                                       self.labels: valid_labels})
                        print('[INFO] Epoch: {}/{}'.format(epoch_i + 1, num_epochs),
                              'Batch: {}/{}'.format(batch_i, num_of_batches),
                              'Train loss: {:.4f}'.format(loss),
                              'Train acc: {:.2f}'.format(train_acc),
                              'Val loss: {:.4f}'.format(val_loss),
                              'Val acc: {:.2f}'.format(val_acc),)
            saver.save(sess, checkpoint_fn)
            print('[INFO] Training weights have been '
                  'successfully saved ats {}.'.format(checkpoint_fn))
            return train_losses, val_losses

    def inference(self, test_features, checkpoint_fn):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_fn)
            print('[INFO] Training weights have been '
                  'successfully loaded from {}.'.format(checkpoint_fn))
            predictions = sess.run(self.predictions,
                                   feed_dict={self.inputs: test_features})
        return predictions
