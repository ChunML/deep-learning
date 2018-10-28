from utils import read_labels_from_file, train_input_fn, eval_input_fn
# from network import Network
from slim_network import vgg16, vgg_arg_scope
import pandas as pd
import os
import tensorflow as tf
import numpy as np
from collections import defaultdict
import itertools
slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string(
    'mode', 'train',
    'running mode, either train or eval'
)

tf.app.flags.DEFINE_float(
    'train_val_rate', 0.8,
    'rate to split dataset to training and validation set'
)

tf.app.flags.DEFINE_string(
    'data_root', './data/all',
    'path to root data'
)

tf.app.flags.DEFINE_integer(
    'num_classes', 120,
    'number of classes'
)

tf.app.flags.DEFINE_string(
    'label_path', './data/all/labels.csv',
    'path to label file'
)

tf.app.flags.DEFINE_integer(
    'batch_size', 64,
    'batch size'
)

tf.app.flags.DEFINE_integer(
    'num_epochs', 10,
    'number of training epochs'
)

tf.app.flags.DEFINE_string(
    'checkpoint_dir', 'models/vgg_16.ckpt',
    'path to checkpoint file'
)

tf.app.flags.DEFINE_string(
    'log_dir', 'models/finetuned_vgg_16.ckpt',
    'path to store new checkpoint file'
)

tf.app.flags.DEFINE_string(
    'exclude_vars', 'vgg_16/fc8',
    'variables to exclude from checkpoint'
)

tf.app.flags.DEFINE_float(
    'lr', 0.001,
    'learning rate'
)

FLAGS = tf.app.flags.FLAGS


def create_train_op(logits, labels):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))
    opt = tf.train.GradientDescentOptimizer(FLAGS.lr)
    train_op = opt.minimize(loss)

    return loss, train_op


def train():
    filenames, labels, int_to_breed, _ = read_labels_from_file(
        FLAGS.data_root, FLAGS.label_path)

    num_train_files = int(len(filenames) * FLAGS.train_val_rate)

    (features, labels), init_op, val_init_op = train_input_fn(
        filenames, labels, FLAGS.batch_size, num_train_files)

    num_batches = num_train_files // FLAGS.batch_size
    val_num_batches = (len(filenames) - num_train_files) // FLAGS.batch_size
    is_training = tf.placeholder(tf.bool)

    with slim.arg_scope(vgg_arg_scope()):
        logits, _ = vgg16(features, FLAGS.num_classes, is_training)

    predictions = tf.argmax(logits, axis=1)
    acc_op = tf.reduce_mean(
        tf.cast(tf.equal(predictions, labels), tf.float32))
    loss_op, train_op = create_train_op(logits, labels)

    variables_to_restore = slim.get_variables_to_restore(exclude=[
                                                         FLAGS.exclude_vars])
    variables_to_initialize = slim.get_variables_to_restore(include=[
                                                            FLAGS.exclude_vars])

    # init_fn = slim.assign_from_checkpoint_fn('models/vgg_16.ckpt', variables_to_restore)

    # slim.learning.train(train_op,
    #                     init_fn=init_fn,
    #                     logdir='logdir',
    #                     number_of_steps=2000,
    #                     log_every_n_steps=10)

    train_info = defaultdict(list)
    with tf.Session() as sess:
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, FLAGS.checkpoint_dir)
        sess.run(tf.variables_initializer(variables_to_initialize))

        saver = tf.train.Saver()

        for epoch_i in range(FLAGS.num_epochs):
            sess.run(init_op)
            losses = []
            accs = []
            for batch_i in range(num_batches):
                _, loss, acc = sess.run(
                    [train_op, loss_op, acc_op], feed_dict={is_training: True})
                losses.append(loss)
                accs.append(acc)
                if batch_i % 10 == 0:
                    print('Epoch: {} Batch: {}/{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                          epoch_i + 1, batch_i + 1, num_batches, loss, acc))

            sess.run(val_init_op)
            val_losses = []
            val_accs = []
            for _ in range(val_num_batches):
                val_loss, val_acc = sess.run(
                    [loss_op, acc_op], feed_dict={is_training: False})
                val_losses.append(val_loss)
                val_accs.append(val_acc)
            loss = np.mean(losses)
            acc = np.mean(accs)
            val_loss = np.mean(val_losses)
            val_acc = np.mean(val_accs)
            train_info['loss'].append(loss)
            train_info['acc'].append(acc)
            train_info['val_loss'].append(val_loss)
            train_info['val_acc'].append(val_acc)
            print('Epoch: {} Loss: {:.4f} Accuracy: {:.4f} Val Loss: {:.4f} Val Accuracy: {:.4f}'.format(
                epoch_i + 1, loss, acc, val_loss, val_acc))

        saver.save(sess, FLAGS.log_dir)

    train_info['loss'] = list(itertools.chain(train_info['loss']))
    train_info['acc'] = list(itertools.chain(train_info['acc']))
    train_df = pd.DataFrame(train_info)
    train_df.to_csv('train_info_{}_epochs.csv'.format(FLAGS.num_epochs))


def evaluate():
    filenames = os.listdir(os.path.join(FLAGS.data_root, 'test'))
    filenames = [os.path.join(FLAGS.data_root, 'test', fn) for fn in filenames]

    features = eval_input_fn(filenames)

    is_training = tf.placeholder(tf.bool)

    with slim.arg_scope(vgg_arg_scope()):
        logits, _ = vgg16(features, FLAGS.num_classes, is_training)
    predictions = tf.argmax(logits, axis=1)
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.log_dir)
        for _ in range(len(filenames)):
            print(sess.run(predictions, feed_dict={is_training: False}))



def main(_):
    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'eval':
        evaluate()
    else:
        print('[ERROR] Mode must either be "train" or "eval"')


if __name__ == '__main__':
    tf.app.run()
