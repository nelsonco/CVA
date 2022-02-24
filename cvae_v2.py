# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
from tensorflow.random import normal
import tensorflow.keras as tk
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as Backend 
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tk.datasets.mnist.load_data()
 

# Normalize and Reshape
x_train, x_test = x_train/255, x_test/255

width_px = x_train.shape[1] 
height_px = x_train.shape[1] 

color_spec = 1 # greyscale

# reshape
x_train = x_train.reshape(x_train.shape[0], height_px, width_px, color_spec)
x_test = x_test.reshape(x_test.shape[0], height_px, width_px, color_spec)


# Input shape of a single image
input_shape = (height_px, width_px, color_spec)

# View an image
plt.figure(1)
plt.imshow(x_train[15])
plt.show()

latent_dim = 2


# Encoder
encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
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

encoder.summary()
decoder.summary()