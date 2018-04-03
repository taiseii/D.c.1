# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implements the Keras Sequential model."""

import itertools
import multiprocessing.pool
import threading
from functools import partial

import keras
import pandas as pd
from keras import backend as K
from keras import layers, models
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras.backend import relu, sigmoid
import numpy as np
import time

from urlparse import urlparse

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter
from keras.applications.resnet50 import ResNet50
from keras.optimizers import RMSprop,Nadam,Adam,Adadelta
import os

#rmsp = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#adadelta = Adadelta(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


def model_fn(labels_dim):
  """Create a Keras Sequential model with layers."""

  model = models.Sequential()
 
  
  model.add(Conv2D(32, kernel_size=(5, 5),
                   activation='relu',
                   padding='same',
                   input_shape=(128, 128, 3)))
  model.add(Conv2D(32, (3, 3),  activation='relu', padding='same'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
  
  model.add(SeparableConv2D(64, (3,3), activation='relu',padding='same',depth_multiplier=4))
  model.add(SeparableConv2D(64, (3,3),activation='relu',padding='same',depth_multiplier=4))
  model.add(MaxPooling2D((2,2),strides=(2,2)))

  model.add(SeparableConv2D(128, (3,3),activation='relu',padding='same',depth_multiplier=4))
  model.add(SeparableConv2D(128, (3,3),activation='relu',padding='same',depth_multiplier=4))
  model.add(MaxPooling2D((2,2),strides=(2,2)))
 
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(labels_dim, activation='softmax'))
  

  compile_model(model)
  return model

def compile_model(model):
  model.compile(loss = keras.losses.categorical_crossentropy,
                optimizer = adam,
                metrics = ['accuracy'])
  return model



def read_train_data():
  #data=np.load("data/trainData.npz") #read from localcomputer
  start_time = time.time()
  print ("Start Read Train Data")
  data = np.load(tf.gfile.Open("gs://retinopathy/data/trainData.npz")) #read from the bucket
  print("Train data read --- %s seconds ---" % (time.time() - start_time))
  X_train=data["X_train"]
  Y_train=data["Y_train"]
  print ("Training - Total examples per class",np.sum(Y_train, axis=0))
  return [X_train,Y_train]

def read_test_data():
  #data=np.load("data/testData.npz") #read from localcomputer
  start_time = time.time()
  print ("Start Read Test Data")
  data = np.load(tf.gfile.Open("gs://retinopathy/data/testData.npz")) #read from the bucket
  print("Test data read --- %s seconds ---" % (time.time() - start_time))
  X_test=data["X_test"]
  Y_test=data["Y_test"]
  print ("Testing - Total examples per class",np.sum(Y_test, axis=0))
  return [X_test,Y_test]
