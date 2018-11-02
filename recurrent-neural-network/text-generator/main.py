import tensorflow as tf
import numpy as np
from collections import namedtuple
import time

with open('anna.txt', 'r') as f:
    text = f.read()

vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

print(text[:100])
print(encoded[:100])
print(len(vocab))


def get_batches(arr, batch_size, n_steps):
    characters_per_batch = batch_size * n_steps
    n_batches = len(arr) // characters_per_batch

    arr = arr[:n_batches * characters_per_batch]
    arr = np.reshape(arr, (batch_size, -1))

    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n + n_steps]
        y = np.zeros(x.shape, dtype=np.int32)
        y[:, :-1] = x[:, 1:]
        yield x, y


batches = get_batches(encoded, 10, 50)
x, y = next(batches)
print(x.shape)
print('x', x[:10, :10])
print('y', y[:10, :10])


def build_inputs(batch_size, num_steps):
    inputs = tf.placeholder(tf.int32, [None, num_steps])
    targets = tf.placeholder(tf.int32, [None, num_steps])

    keep_prob = tf.placeholder(tf.float32)

    return inputs, targets, keep_prob


def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    def build_cell(lstm_size, keep_prob):
        lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop

    cell = tf.nn.rnn_cell.MultiRNNCell(
        [build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state


def build_output(lstm_output, lstm_size, num_classes):
    seq_output = tf.concat(lstm_output, axis=1)
    x = tf.reshape(seq_output, [-1, lstm_size])

    logits = tf.layers.dense(x, num_classes,
                             kernel_initializer=tf.initializers.truncated_normal(stddev=0.01))
    out = tf.nn.softmax(logits)

    return out, logits


def build_loss(logits, targets, lstm_size, num_classes):
    y_one_hot = tf.one_hot(targets, depth=num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_reshaped,
        logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    return loss


def build_optimizer(loss, learning_rate, grad_clip):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    return train_op


class CharRNN:
    def __init__(self, num_classes, batch_size=64,
                 num_steps=50, lstm_size=128,
                 num_layers=2, learning_rate=1e-3,
                 grad_clip=5, sampling=False):
        if sampling:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()

        self.inputs, self.targets, self.keep_prob = build_inputs(
            batch_size, num_steps)

        cell, self.initial_state = build_lstm(
            lstm_size, num_layers, batch_size, self.keep_prob)

        x_one_hot = tf.one_hot(self.inputs, depth=num_classes)

        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot,
                                           initial_state=self.initial_state)
        self.final_state = state

        self.prediction, self.logits = build_output(
            outputs, lstm_size, num_classes)

        self.loss = build_loss(self.logits, self.targets,
                               lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)


batch_size = 10
num_steps = 50
lstm_size = 128
num_layers = 2
learning_rate = 0.01
keep_prob = 0.5

num_epochs = 20
print_every_n = 50
save_every_n = 200
model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers,
                learning_rate=learning_rate)
saver = tf.train.Saver(max_to_keep=100)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    counter = 0
    for e in range(num_epochs):
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss,
                                                 model.final_state,
                                                 model.optimizer],
                                                feed_dict=feed)
            if counter % print_every_n == 0:
                end = time.time()
                print('Epoch: {}/{}...'.format(e+1, num_epochs),
                      'Training steps: {}...'.format(counter),
                      'Training loss: {:.4f}...'.format(batch_loss),
                      '{:.4f} s/batch'.format(end-start))

        saver.save(sess, 'models/model_at_epoch_{}.ckpt'.format(e + 1))
    saver.save(sess, 'models/model_final.ckpt')
