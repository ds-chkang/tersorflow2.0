# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:03:35 2022

@author: dschk
"""

import os
import json
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation
from tensorflow.keras.layers import InputLayer

### To prevent tensorflow from eating up GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

model = Sequential()
#model.add(InputLayer(input_shape=(28, 28, 1)))
model.add(Flatten(name='flatter'))
model.add(Dense(units=10, name='dense_1'))
model.add(Activation('relu', name='dense_a_act'))
model.add(Dense(units=2, name='dense_2'))
model.add(Activation('softmax', name='dense_2_act'))

#model.build()
model.build(input_shape=(None, 28, 28, 1))

model.summary()
print(model.layers[0].get_weights())


## mode 초기화
#tf.keras.backend.clear_session()

# %%
class TestModel(Model):
    def __init__(self):
        super(TestModel, self).__init__()
        
        self.flatten = Flatten()
        self.d1 = Dense(units=10)
        self.d1_act = Activation('relu')
        self.d2 = Dense(units=2)
        self.d2_act = Activation('softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d1_act(x)
        x = self.d2(x)
        x = self.d2_act(x)
        return x
        
model = TestModel()

# 모델 서브클래싱을 할 때는 아래와 같이 input_shape을 build에 정의 해야 됨.    
model.build(input_shape=(None, 28, 28, 1))
model.summary()
    

print(model.layers[0].get_weights())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    