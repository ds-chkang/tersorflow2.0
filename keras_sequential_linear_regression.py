# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:52:11 2022

@author: dschk
"""

import tensorflow as tf
import matplotlib.pyplot as plt

x_train = tf.random.normal(shape=(1000, ), dtype=tf.float32)
y_train = 3*x_train + 1 + 0.2*tf.random.normal(shape=(1000, ), dtype=tf.float32)

x_test = tf.random.normal(shape=(300,), dtype=tf.float32)
y_test = 3*x_test + 1 + 0.2*tf.random.normal(shape=(300,), dtype=tf.float32)

model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=1, activation='linear')
    ])

model.compile(loss='mean_squared_error', optimizer='SGD')
model.fit(x_train, y_train, epochs=50, verbose=2)
model.evaluate(x_test, y_test, verbose=1)