# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:26:16 2022

@author: dschk
"""

import tensorflow as tf
import numpy as np

train_x = np.arange(20).astype(np.float32).reshape(-1,1)
train_y = 3*train_x+1

train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_ds = train_ds.shuffle(100).batch(32)

for x, y in train_ds:
    print(x.shape, y.shape, '\n')
    
# %%
from tensorflow.keras.datasets import mnist
from tensorflow.data import Dataset
import matplotlib.pyplot as plt

((train_images, train_labels), (test_images, test_labels))=mnist.load_data()

train_ds = Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.shuffle(60000).batch(12)

test_ds = Dataset.from_tensor_slices((test_images,test_labels))
test_ds = test_ds.batch(12)
    
train_ds_iter = iter(train_ds)
images, labels = next(train_ds_iter)

fig, axes = plt.subplots(4,3, figsize=(10,10))

for ax_idx, ax in enumerate(axes.flat):
    image = images[ax_idx, ...]
    label = labels[ax_idx]

    ax.imshow(image.numpy(), 'gray')
    ax.set_title(label.numpy(), fontsize=20)    
    
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    
    
    
    
    
    
    
    
    