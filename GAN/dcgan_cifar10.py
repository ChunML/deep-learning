import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob
import os

BUFFER_SIZE = 50000
BATCH_SIZE = 128
LR = 2e-4
BETA1 = 0.5
EPOCHS = 100
NOISE_DIM = 100
NUM_FAKE_IMAGES = 16

cifar, info = tfds.load('cifar10', with_info=True, as_supervised=True)
train_data, test_data = cifar['train'], cifar['test']
train_data = train_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_data = test_data.batch(BATCH_SIZE)

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(100,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Reshape((4, 4, 512)),
            tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        ])
  
    def call(self, z, training=True):
        return self.model(z, training)
        
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = tf.keras.Sequential([
            # tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 3]),
            # tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ])
  
    def call(self, image, training=True):
        return self.model(image, training)
        
generator = Generator()
discriminator = Discriminator()

loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_pred, fake_pred):
    real_loss = loss_func(tf.ones_like(real_pred), real_pred)
    fake_loss = loss_func(tf.zeros_like(fake_pred), fake_pred)
    total_loss = real_loss + fake_loss
    return total_loss
    
def generator_loss(fake_pred):
    return loss_func(tf.ones_like(fake_pred), fake_pred)
  
generator_opt = tf.keras.optimizers.Adam(LR, beta_1=BETA1)
discriminator_opt = tf.keras.optimizers.Adam(LR, beta_1=BETA1)

seed = tf.random.normal([NUM_FAKE_IMAGES, NOISE_DIM])

@tf.function
def train_step(images, noise):  
    # One tape can only keep track of one gradient flow
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        fake_images = generator(noise, training=True)
    
        real_outputs = discriminator(images, training=True)
        fake_outputs = discriminator(fake_images, training=True)
    
        gen_loss = generator_loss(fake_outputs)
        dis_loss = discriminator_loss(real_outputs, fake_outputs)
    
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    dis_gradients = dis_tape.gradient(dis_loss, discriminator.trainable_variables)
  
    generator_opt.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_opt.apply_gradients(zip(dis_gradients, discriminator.trainable_variables))

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    print_images = tf.dtypes.cast(predictions * 127.5 + 127.5, tf.int32)
  
    fig = plt.figure(figsize=(4, 4))
  
    for i in range(print_images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(print_images[i, :, :, :], cmap='gray')
        plt.axis('off')
  
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

def train(dataset, epochs):
    for epoch in range(epochs):
        for images, _ in dataset.take(-1):
            noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
            images = (tf.dtypes.cast(images, tf.float32) - 127.5) / 127.5
            train_step(images, noise)
    
        generate_and_save_images(generator, epoch + 1, seed)

    generate_and_save_images(generator, epochs, seed)
    
train(train_data, EPOCHS)

# Create GIF image
with imageio.get_writer('cifar10-gan.gif', mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
  
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
