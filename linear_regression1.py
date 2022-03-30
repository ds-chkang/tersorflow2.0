# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = tf.random.normal(shape=(100, 1), dtype=tf.float32)
y_data = 3*x_data + 1

print(x_data.shape)
print(y_data.shape)

w = tf.Variable(-1.)
b = tf.Variable(-1.)

learning_rate = 0.01
EPOCHS = 10
w_trace, b_trace = [], []
for epoch in range(EPOCHS):
    for x, y in zip(x_data, y_data):
        with tf.GradientTape() as tape:
            prediction = w*x + b
            loss = (prediction - y)**2
            
        gradients = tape.gradient(loss, [w, b])

        w_trace.append(w.numpy())
        b_trace.append(b.numpy())
        w = tf.Variable(w - learning_rate*gradients[0])
        b = tf.Variable(b - learning_rate*gradients[1])
        
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(w_trace, label='weight')
ax.plot(b_trace, label='bias')
ax.legend(fontsize=20)


        
 
        












