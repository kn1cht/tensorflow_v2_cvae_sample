#!/usr/bin/env python
# coding: utf-8
#
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/cvae.ipynb
#   Copyright 2020 The TensorFlow Authors. All rights reserved.
#   distributed in the Apache License 2.0 (https://github.com/tensorflow/docs/blob/master/LICENSE)
#   contributed by @lamberta @MarkDaoust @yashk2810
#   changed by @kn1cht (June 15, 2020)

import numpy as np
import tensorflow as tf

class CVAE(tf.keras.Model):
  """Conditional variational autoencoder."""

  def __init__(self, latent_dim, label_size):
    super(CVAE, self).__init__()
    (self.latent_dim, self.label_size) = (latent_dim, label_size)
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, label_size + 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim + label_size,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None, y=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, y, apply_sigmoid=True)

  def encode(self, x, y):
    n_sample = x.shape[0]
    image_size = x.shape[1:3]

    y_onehot = tf.reshape(tf.one_hot(y, self.label_size), [n_sample, 1, 1, self.label_size]) # 1 x 1 x label_size
    k = tf.ones([n_sample, *image_size, 1]) # {image_size} x 1
    h = tf.concat([x, k * y_onehot], 3) # {image_size} x (1 + label_size)

    mean, logvar = tf.split(self.encoder(h), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, y=None, apply_sigmoid=False):
    n_sample = z.shape[0]
    if not y is None:
      y_onehot = tf.reshape(tf.one_hot(y, self.label_size), [n_sample, self.label_size]) # label_size
      h = tf.concat([z, y_onehot], 1) # latent_dim + label_size
    else:
      h = tf.concat([z, tf.zeros([n_sample, self.label_size])], 1)  # latent_dim + label_size
    logits = self.decoder(h)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits


optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def compute_loss(model, xy):
  (x, y) = xy # x: image, y: label
  mean, logvar = model.encode(x, y)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z, y)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, xy, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, xy)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

