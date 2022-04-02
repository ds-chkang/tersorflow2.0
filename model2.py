# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 18:46:00 2022

@author: dschk
"""

import numpy as np
from termcolor import colored

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.metrics import CategoricalAccuracy

x_data = np.array([
    [1,2,0], 
    [5,4,3], 
    [1,2,-1], 
    [3,1,0], 
    [2,4,2],
    [4,1,2], 
    [-1,3,2], 
    [4,3,3], 
    [0,2,6], 
    [2,2,1],
    [1,-2,-2], 
    [0,1,3], 
    [1,1,3], 
    [0,1,4], 
    [2,3,3]]).astype(np.float32).reshape(15, 3)

y_data = np.array([-4, 4, -6, 3, -4, 9, -7, 5, 6, 0, 4, 3, 5, 5, 1]).astype(np.float32).reshape(15,1)

class LinearRegression(Model):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.d1 = Dense(units=1, input_shape=(3,), activation='linear')
    
    def call(self, x):
        predictions = self.d1(x)
        return predictions
    
    
LR = 0.01
EPOCHS = 100

model = LinearRegression()
loss_object = MeanSquaredError()
optimizer = SGD(LR)

loss_metric = Mean()


for epoch in range(EPOCHS):
    for x, y in zip(x_data, y_data):
        x = tf.reshape(x, (3, 1))
        y = tf.reshape(y, (1, 1))
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = (predictions - y)**2
            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_metric(loss)

    print(colored('Epoch: ', 'cyan', 'on_white'), epoch+1)
    template = 'Train Loss: {:.4f}'

    ds_loss = loss_metric.result()

    print(template.format(ds_loss))

    loss_metric.reset_states()


model.summary()


print(model.input)
print(model.output)

model_weights = model.weights
print(len(model_weights))
for model_weight in model_weights:
    print(model_weight.numpy())















    