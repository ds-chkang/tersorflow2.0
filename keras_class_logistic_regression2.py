# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 20:49:35 2022

@author: dschk
"""

from termcolor import colored

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Mean

n_train = 1000

train_x = np.random.normal(0, 1, size=(n_train, 1)).astype(np.float32)
train_x_noise = train_x + 0.2*np.random.normal(0, 1, size=(n_train, 1))

train_y = (train_x_noise > 0).astype(np.int32)

train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_ds = train_ds.shuffle(n_train).batch(8)

model = Sequential()
model.add(Dense(units=2, activation='softmax'))

loss_object = SparseCategoricalCrossentropy()
optimizer = SGD(learning_rate=1)

train_loss = Mean()
train_acc = SparseCategoricalAccuracy()

EPOCHS = 10

for epoch in range(EPOCHS):
    for x, y in train_ds:
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_acc(y, predictions)

    print(colored('Epoch: ', 'red', 'on_white'), epoch+1)
    template = 'Train Loss: {:.4f}\t Train Accuracy: {:.2f}%\n'
    print(template.format(train_loss.result(), train_acc.result()*100))
    
    train_loss.reset_states()
    train_acc.reset_states()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        