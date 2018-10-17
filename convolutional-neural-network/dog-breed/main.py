from utils import read_labels_from_file, input_fn
from network import Network
import pandas as pd
import os
import tensorflow as tf
import numpy as np
from collections import defaultdict

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
train_val_rate = 0.8
label_path = './data/all/labels.csv'
batch_size = 64
num_epochs = 100
num_epochs_per_decay = 20


def main():
    filenames, labels, int_to_breed, breed_to_int = read_labels_from_file(
        label_path)
    num_train_files = int(len(filenames) * train_val_rate)

    (features, labels), init_op, val_init_op = input_fn(
        filenames, labels, batch_size, num_train_files)

    num_batches = num_train_files // batch_size
    val_num_batches = (len(filenames) - num_train_files) // batch_size

    decay_steps = num_epochs_per_decay * num_batches

    net = Network(features, labels, len(int_to_breed),
                  initial_learning_rate=0.01, decay_steps=decay_steps)

    train_info = defaultdict(list)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(num_epochs):
            sess.run(init_op)
            for batch_i in range(num_batches):
                _, loss, acc = sess.run([net.opt, net.loss, net.accuracy], feed_dict={
                                        net.is_training: True})
                train_info['loss'].append(loss)
                train_info['acc'].append(acc)
                print('Epoch: {} Batch: {}/{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                    epoch_i + 1, batch_i + 1, num_batches, loss, acc))

            sess.run(val_init_op)
            val_losses = []
            val_accs = []
            for _ in range(val_num_batches):
                val_loss, val_acc = sess.run([net.loss, net.accuracy], feed_dict={
                    net.is_training: False})
                val_losses.append(val_loss)
                val_accs.append(val_acc)
            val_loss = np.mean(val_losses)
            val_acc = np.mean(val_accs)
            print('Epoch: {} Loss: {:.4f} Accuracy: {:.4f} Val Loss: {:.4f} Val Accuracy: {:.4f}'.format(
                epoch_i + 1, batch_i, num_batches, loss, acc, val_loss, val_acc))

    train_df = pd.DataFrame(train_info)
    train_df.to_csv('train_info_{}_epochs.csv'.format(num_epochs))


if __name__ == '__main__':
    main()
