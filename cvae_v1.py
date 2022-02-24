# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from IPython import display

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from cvae_main import CVAE
import cvae_main as main
import time



# Parameters
num_train = 60000
num_test = 10000

epochs = 3
lr = 1e-4
batch_size = 32

latent_dim = 2

optimizer = tf.optimizers.Adam(learning_rate=lr)

# Load MNIST Data
[train_dataset, test_dataset, (x_train, y_train),
 (x_test, y_test), input_shape] = main.loadMNIST(num_train, num_test, batch_size)

# Initialize CVAE
CVAE = CVAE(latent_dim, input_shape)

# Begin Training
for epoch in range(1, epochs + 1):
    start_time = time.time()
    for image_batch in train_dataset:
        main.train_step(CVAE, image_batch, optimizer)
    end_time = time.time()
    
    mean_loss = tf.keras.metrics.Mean()
    for image_batch in test_dataset:
        mean_loss(main.compute_loss(CVAE, image_batch))
    elbo = -mean_loss.result() # EVIDENCE LOWER BOUND
    display.clear_output(wait=False)
    print('Epoch: {}, Test Set Evidence Lower Bound: {}, time elapse for current epoch: {}'
          .format(epoch, elbo, end_time-start_time)          
          )
    
        



