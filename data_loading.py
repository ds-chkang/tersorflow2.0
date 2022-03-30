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
predictions = np.array([0.3, 0.7]).reshape(-1, 1)
labels = np.array([1, 0]).reshape(-1, 1)

loss = loss_object(labels, predictions)
loss_manual = -1*(labels*np.log(predictions) + (1-labels)*np.log(1-predictions))

loss_manual = np.mean(loss_manual)

print(loss.numpy())
print(loss_manual)







































