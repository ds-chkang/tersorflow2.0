# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:42:31 2022

@author: dschk
"""

import tensorflow as tf
import numpy as np

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

loss_object = BinaryCrossentropy()

predictions = np.array([0.3]).reshape(-1, 1)
labels = np.array([1])

loss = loss_object(labels, predictions)
loss_manual = -1*(labels*np.log(predictions) + (1-labels)*np.log(1-predictions))

print(loss.numpy())
print(loss_manual)


# %%
predictions = np.array([0.3, 0.6]).reshape(-1, 1)
labels = np.array([1, 0]).reshape(-1, 1)

loss = loss_object(labels, predictions)
loss_manual = -1*(labels*np.log(predictions) + (1-labels)*np.log(1-predictions))

loss_manual = np.mean(loss_manual)

print(loss.numpy())
print(loss_manual)

# %%
predictions = np.array([[0.3, 0.7], [0.4, 0.6], [0.1, 0.9]])
labels = np.array([[0, 1], [1, 0], [1, 0]])

loss = loss_object(labels, predictions)

loss_manual = -1*labels*np.log(predictions)
loss_manual = np.sum(loss_manual, axis=1)
loss_manual = np.mean(loss_manual)

print(loss.numpy())
print(loss_manual)

# %%

loss_object = CategoricalCrossentropy()

predictions = np.array([[0.2, 0.1, 0.7], [0.4, 0.3, 0.3], [0.1, 0.8, 0.1]])
labels = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

#print(predictions)
#print(labels)

loss = loss_object(labels, predictions)

loss_manual = -1*(labels*np.log(predictions))

loss_manual = np.sum(loss_manual, axis=1)
loss_manual = np.mean(loss_manual)

print(loss.numpy())
print(loss_manual)

# %%
loss_object = SparseCategoricalCrossentropy()

predictions = np.array([[0.2, 0.1, 0.7], [0.4, 0.3, 0.3], [0.1, 0.8, 0.1]])
labels = np.array([2, 1, 0])

loss = loss_object(tf.constant(labels), tf.constant(predictions))

ce_loss = 0
for data_idx in range(len(labels)):
    prediction = predictions[data_idx]
    label = labels[data_idx]
    
    t_prediction = prediction[label]
    ce_loss += -1*np.log(t_prediction)
    
ce_loss = ce_loss/len(labels)

print(loss.numpy())
print(loss_manual)

# %%
metric = CategoricalAccuracy()

predictions = np.array([[0.2, 0.2, 0.6], [0.1, 0.8, 0.1]])
labels = np.array([[0, 0, 1], [0, 0, 1]])

acc = metric(labels, predictions)
print((acc*100).numpy())


# %%
metric = SparseCategoricalAccuracy()

predictions = np.array([[0.2, 0.2, 0.6], [0.1, 0.8, 0.1]])
labels = np.array([2, 1])

acc = metric(labels, predictions)
print((acc*100).numpy())



















































