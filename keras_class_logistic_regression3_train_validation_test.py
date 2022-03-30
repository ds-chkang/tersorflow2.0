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

n_train, n_validation, n_test = 1000, 300, 300

train_x = np.random.normal(0, 1, size=(n_train, 1)).astype(np.float32)
train_x_noise = train_x + 0.2*np.random.normal(0, 1, size=(n_train, 1))
train_y = (train_x_noise > 0).astype(np.int32)

validation_x = np.random.normal(0, 1, size=(n_validation, 1)).astype(np.float32)
validation_x_noise = validation_x + 0.2*np.random.normal(0, 1, size=(n_validation, 1))
validation_y = (validation_x_noise > 0).astype(np.int32)

test_x = np.random.normal(0, 1, size=(n_test, 1)).astype(np.float32)
test_x_noise = test_x + 0.2*np.random.normal(0, 1, size=(n_test, 1))
test_y = (test_x_noise > 0).astype(np.int32)

train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_ds = train_ds.shuffle(n_train).batch(8)

validation_ds = tf.data.Dataset.from_tensor_slices((validation_x, validation_y))
validation_ds = validation_ds.batch(n_validation)

test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_ds = test_ds.batch(n_test)

model = Sequential()
model.add(Dense(units=2, activation='softmax'))

loss_object = SparseCategoricalCrossentropy()
optimizer = SGD(learning_rate=1)

train_loss = Mean()
train_acc = SparseCategoricalAccuracy()

validation_loss = Mean()
validation_acc = SparseCategoricalAccuracy()

test_loss = Mean()
test_acc = SparseCategoricalAccuracy()

EPOCHS = 10

train_losses, validation_losses = [], []
train_accs, validation_accs = [], []

for epoch in range(EPOCHS):
    for x, y in train_ds:
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_acc(y, predictions)

    # validate model
    for x, y in validation_ds:
        predictions = model(x)
        loss = loss_object(y, predictions)
    
        validation_loss(loss)
        validation_acc(y, predictions)


    print(colored('Epoch: ', 'red', 'on_white'), epoch+1)
    template = 'Train Loss: {:.4f}\t Train Accuracy: {:.2f}%\n' + \
        'Validation Loss: {:.4f}\t Validation Accuracy: {:.2f}%\n'
    print(template.format(train_loss.result(), 
                          train_acc.result()*100,
                          validation_loss.result(),
                          validation_acc.result()*100))
    
    train_losses.append(train_loss.result())
    train_accs.append(train_acc.result()*100)
    validation_losses.append(validation_loss.result)
    validation_accs.append(validation_acc.result()*100)
    
    train_loss.reset_states()
    train_acc.reset_states()
    validation_loss.reset_states()
    validation_acc.reset_states()
    
# test model
for x, y in test_ds:
    predictions = model(x)
    loss = loss_object(y, predictions)
    
    test_loss(loss)
    test_acc(y, predictions)
    
print(colored('Final Result: ', 'cyan', 'on_white'))
template = 'Test Loss: {:.4f}\t Test Accuracy: {:.2f}%\n'
print(template.format(test_loss.result(), test_acc.result()*100))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        