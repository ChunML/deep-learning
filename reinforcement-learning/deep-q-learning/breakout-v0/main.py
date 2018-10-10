import tensorflow as tf
from memory import Memory
from network import QNetwork
import gym
import numpy as np
import cv2
import PIL

env = gym.make('Breakout-v0')

# Number of training episodes
train_episodes = 10000

# Number of steps per episode
max_steps = 200

# Future reward discount
gamma = 0.99

# Exploration parameters
# Exploration probability at start
explore_start = 1.0

# Minumum exploration probability
explore_stop = 0.1

# Exploration decay rate
decay_rate = 0.0001

# Network hyper parameters
# Number of units in FC layer
hidden_size = 256

# Learning rate for Q-network
learning_rate = 0.0002

# Memory parameters
# Memory capacity
memory_size = 100000

# Experience mini-batch size
batch_size = 32

# Number of experiences to prepare
pretrain_length = batch_size


def get_state():
    state = env.render(mode='rgb_array')
    state = np.ascontiguousarray(state, dtype=np.float32) / 255.
    return resize(state, (84, 84))


def resize(img, size, interpolation=PIL.Image.NEAREST):
    if interpolation == PIL.Image.NEAREST:
        cv_interpolation = cv2.INTER_NEAREST
    elif interpolation == PIL.Image.BILINEAR:
        cv_interpolation = cv2.INTER_LINEAR
    elif interpolation == PIL.Image.BICUBIC:
        cv_interpolation = cv2.INTER_CUBIC
    elif interpolation == PIL.Image.LANCZOS:
        cv_interpolation = cv2.INTER_LANCZOS4
    H, W = size
    img = cv2.resize(img, dsize=(W, H), interpolation=cv_interpolation)
    # If input is a grayscale image, cv2 returns a two-dimentional array.
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    return img


def train():
    env.reset()

    _, reward, done, _ = env.step(env.action_space.sample())
    state = get_state()

    memory = Memory(max_size=memory_size)

    for _ in range(pretrain_length):
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        next_state = get_state()

        if done:
            next_state = np.zeros(state.shape)
            memory.add((state, action, reward, next_state))

            env.reset()

            _, reward, done, _ = env.step(env.action_space.sample())
            state = get_state()
        else:
            memory.add((state, action, reward, next_state))
            state = next_state

    img_shape = state.shape
    network = QNetwork(height=img_shape[0],
                       width=img_shape[1],
                       channel=img_shape[2],
                       learning_rate=learning_rate)
    saver = tf.train.Saver()
    save_file = 'checkpoints/cartpole.ckpt'
    rewards_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        for ep in range(1, train_episodes + 1):
            total_reward = 0
            t = 0
            while t < max_steps:
                step += 1
                env.render()

                explore_p = explore_stop + \
                    (explore_start - explore_stop) * np.exp(-decay_rate * step)
                if explore_p > np.random.rand():
                    action = env.action_space.sample()
                else:
                    feed = {network.inputs_: state.reshape((1, *state.shape))}
                    Qs = sess.run(network.output, feed_dict=feed)
                    action = np.argmax(Qs)

                _, reward, done, _ = env.step(action)
                next_state = get_state()
                total_reward += reward

                if done:
                    next_state = np.zeros(state.shape)
                    t = max_steps

                    print('Episode: {}'.format(ep),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore Prob: {:.4f}'.format(explore_p))
                    rewards_list.append((ep, total_reward))

                    memory.add((state, action, reward, next_state))
                    env.reset()
                    _, reward, done, _ = env.step(env.action_space.sample())
                    state = get_state()

                else:
                    memory.add((state, action, reward, next_state))
                    state = next_state
                    t += 1

                batch = memory.sample(batch_size)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])

                target_Qs = sess.run(network.output, feed_dict={
                                     network.inputs_: next_states})

                temp_shape = next_states.shape
                is_episode_over = (next_states.reshape((temp_shape[0], -1)) ==
                                   np.zeros((temp_shape[1] * temp_shape[2] * temp_shape[3]))).all(axis=1)
                target_Qs[is_episode_over] = (0, 0, 0, 0)

                targets = rewards + gamma * np.max(target_Qs, axis=1)

                loss, _ = sess.run([network.loss, network.opt],
                                   feed_dict={network.inputs_: states,
                                              network.targetQs_: targets,
                                              network.actions_: actions})

        saver.save(sess, save_file)


if __name__ == '__main__':
    train()
