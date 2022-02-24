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

encoder = tk.Sequential(
    [
     
      Input(shape=input_shape, name='encoder_input'), # 28, 28, 1
      Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', # number of filters is the number of feature maps
             padding='same', name='hidden1'), # why not include padding
      Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', 
             padding='same', name="hidden2"),
      Flatten(name='flatten'),
      Dense(2*latent_dim, name='mean-log_variance'),
     ]
    
    )
encoder_summary = encoder.summary()


# Reparamaterize 
def reparameterize(mean, logvar):
    epsilon = normal(shape=mean.shape) # 2d normal distribution?
    return epsilon*tf.exp(logvar/2)+mean


# Decoder
num_conv2d = 2
stride = 2
reshape_dec = (int(input_shape[0]/(stride*num_conv2d)), int(input_shape[1]/(stride*num_conv2d)),32)
decoder = tk.Sequential(
    [
     Input(shape=(latent_dim, ), name='decoder_input'),
     Dense(units=reshape_dec[0]*reshape_dec[1]*reshape_dec[2], activation='relu'),
     Reshape(target_shape=(reshape_dec[0], reshape_dec[1], reshape_dec[2])),
     Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
     Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
     Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
     ]
    )

decoder_summary = decoder.summary()


# Sample - Reparamaterize
