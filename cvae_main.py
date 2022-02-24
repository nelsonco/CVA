# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:50:35 2022

@author: Court
"""
import tensorflow as tf
from tensorflow.random import normal
import tensorflow.keras as tk
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.nn import sigmoid_cross_entropy_with_logits 
from tensorflow.data import Dataset
import numpy as np

def loadMNIST(num_train, num_test, batch_size):
    # Load Data
    (x_train, y_train), (x_test, y_test) = tk.datasets.mnist.load_data()
     
    
    # Normalize and Reshape
    x_train, x_test = x_train/255, x_test/255
    
    width_px = x_train.shape[1] 
    height_px = x_train.shape[1] 
    
    color_spec = 1 # greyscale
    
    # reshape
    x_train = x_train.reshape(x_train.shape[0], height_px, width_px, color_spec)
    x_test = x_test.reshape(x_test.shape[0], height_px, width_px, color_spec)
    
    # NOT SURE
    x_train = np.where(x_train > .5, 1.0, 0.0).astype('float32')
    x_test = np.where(x_test > .5, 1.0, 0.0).astype('float32')

    
    # Input shape of a single image
    input_shape = (height_px, width_px, color_spec)
    
    # Batch and Shuffle
    train_dataset = (Dataset.from_tensor_slices(x_train).shuffle(num_train).batch(batch_size))
    test_dataset = (Dataset.from_tensor_slices(x_test).shuffle(num_test).batch(batch_size))
    
    return train_dataset, test_dataset, (x_train, y_train), (x_test, y_test), input_shape

class CVAE(Model):
    """ Convolutional Variational Autoencoder"""
    def __init__(self, latent_dim, img_shape):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim   
        self.img_shape = img_shape
        # initialize encoder
        self.encoder = tk.Sequential(
            [
              Input(shape=self.img_shape, name='encoder_input'), # 28, 28, 1
              Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', 
                     padding='same', name='conv1'), # why not include padding
              Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', 
                     padding='same', name="conv2"),
              Flatten(name='flatten'),
              Dense(2*latent_dim, name='mean-log_variance'),
             ], 
            
            name='ENCODER'
            
            )

        # parameters for decoder
        stride = self.encoder.get_layer('conv1').strides[0]
        num_conv2d = 2
        reshape_dec = (int(self.img_shape[0]/(stride*num_conv2d)),
                       int(self.img_shape[1]/(stride*num_conv2d)),32)
        
        # initialize decoder
        self.decoder = tk.Sequential(
            [
             Input(shape=(latent_dim, ), name='decoder_input'),
             Dense(units=reshape_dec[0]*reshape_dec[1]*reshape_dec[2],
                   activation='relu', name='dense'),
             Reshape(target_shape=(reshape_dec[0], reshape_dec[1], 
                   reshape_dec[2]), name='reshape'),
             Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                   padding='same', activation='relu', name='convT1'),
             Conv2DTranspose(filters=32, kernel_size=3, strides=2, 
                   padding='same', activation='relu', name='convT2'),
             Conv2DTranspose(filters=1, kernel_size=3, strides=1, 
                   padding='same', name='convT3'),
             ], 
            
            name='DECODER'
            
            )
    
    
    ### ENCODER FUNCTIONS
    
    # Encode
    @tf.function
    def mean_logvar(self, input_image):
        mean, logvar = tf.split(self.encoder(input_image), 
                                num_or_size_splits=2, axis=1)
        return mean, logvar
    
    # Reparamaterize 
    def z(self, mean, logvar):
        epsilon = normal(shape=mean.shape) # 2d normal distribution?
        return mean + epsilon*tf.exp(logvar/2)
    
    def encode(self, input_image):
        mean, log_var = self.mean_logvar(input_image)
        z = self.z(mean, log_var)
        return z, mean, log_var
    
    ### DECODER FUNCTIONS
    
    def decode(self, z, apply_sigmoid=False):
        reconstructed = self.decoder(z)
        if apply_sigmoid:
            probability_dist = tf.sigmoid(reconstructed)
            return probability_dist
        return reconstructed
    
    def generate_new_from_latent(self, epsilon=None):
        if epsilon is None:
            epsilon = tf.random.normal(shape=(100, self.latent_dim)) # Why 100
        return self.decode(epsilon, apply_sigmoid=True)
    
### LOSS

def log_gaussian_dist(sample, mean, logvar, raxis=1):
    # gaussian_dist = (1/(logvar*np.sqrt(2*np.pi)))*np.exp((-.5*(sample-mean)**2)/(logvar**2))
    # return np.log(gaussian_dist)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + np.log(2.*np.pi)),
      axis=raxis)

def compute_loss(model, input_img):
    z, mean, logvar = model.encode(input_img)
    x_reconstructed = model.decode(z)
    # measures the probability error in classes with two outcomes
    # confusing
    cross_entropy = sigmoid_cross_entropy_with_logits(logits=x_reconstructed, labels=input_img)
    logpx_z = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
    logpz = log_gaussian_dist(z, 0., 0.)
    logqz_x = log_gaussian_dist(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    
### TRAIN MODEL
@tf.function
def train_step(model,input_img,optimizer):
    
    with tf.GradientTape() as tape:
        loss = compute_loss(model, input_img)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

