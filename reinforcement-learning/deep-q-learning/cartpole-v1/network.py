import tensorflow as tf

class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4,
                 action_size=2, hidden_size=10,
                 name='QNetwork'):
        with tf.variable_scope(name):
            # states as network input
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')

            # chosen actions to compute Q values
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)

            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target_Q')

            # Hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)

            # Output layer
            self.output = tf.contrib.layers.fully_connected(self.fc2, action_size,
                                                            activation_fn=None)
            
            # Q values from network
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)

            # Compute training loss
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))

            # Training opt
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)