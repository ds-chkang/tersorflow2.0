# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:34:42 2022

@author: dschk
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

x_train = tf.random.normal(shape=(10,), dtype=tf.float32)
y_train = 3*x_train + 1+ 0.2*tf.random.normal(shape=(10, ), dtype=tf.float32)

x_test = tf.random.normal(shape=(3, ), dtype=tf.float32)
y_test = 3*x_test + 1 + 0.2*tf.random.normal(shape=(3, ), dtype=tf.float32)

class LinearPredictor(tf.keras.Model):
    def __init__(self):
        super(LinearPredictor, self).__init__()
        self.d1 = tf.keras.layers.Dense(units=1, activation='linear')
        
    def call(self, x):
        x = self.d1(x)
        return x
    
EPOCHS= 10
LR = 0.01
model = LinearPredictor()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate = LR)

for epoch in range(EPOCHS):
    for x, y in zip(x_train, y_train):
        x = tf.reshape(x, (1,1))
        with tf.GradientTape() as tape:
            prediction = model(x)
            loss = loss_object(y, prediction)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
    print(colored('Epoch: ', 'red', 'on_white'), epoch+1)
      
    template = 'Train Loss: {:.4f}\n'
    print(template.format(loss))
    