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

"""Experimentation with wide residual neural network(WRNN)"""
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
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras.backend import relu, sigmoid
import numpy as np
np.random.seed(2**10)
import time
from keras.optimizers import SGD
from keras.regularizers import 12

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from urlparse import urlparse

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter
import os

import sys
sys.stdout=sys.stderr
sys.setrecursionlimit(2**20)

sgd = SGD(lr=0.1,momentum=0.9, nesterov=True) 
img_size = 128
k=10
dropout_p=0
weidth_decay=0.0005
      #sgd = SGD(lr=0.1,momentum=0.9, nesterov=True) #read SGD optimizer keras
use_bias = False        # following functions 'FCinit(model)' and 'DisableBias(model)' in utils.lua
weight_init="he_normal"
nb_classes=5

if K.image_dim_ordering() == "th":
    logging.debug("image_dim_ordering = 'th'")
    channel_axis = 1
    input_shape = (3, image_size, image_size)
else:
    logging.debug("image_dim_ordering = 'tf'")
    channel_axis = -1
    input_shape = (image_size, image_size, 3)

def _wide_basic(n_input_plane, n_output_plane, stride):
    def f(net):
        # format of conv_params:
        #               [ [nb_col="kernel width", nb_row="kernel height",
        #               subsample="(stride_vertical,stride_horizontal)",
        #               border_mode="same" or "valid"] ]
        # B(3,3): orignal <<basic>> block
        conv_params = [ [3,3,stride,"same"],
                        [3,3,(1,1),"same"] ]
        
        n_bottleneck_plane = n_output_plane

        # Residual block
        for i, v in enumerate(conv_params):
            if i == 0:
                if n_input_plane != n_output_plane:
                    net = BatchNormalization(axis=channel_axis)(net)
                    net = Activation("relu")(net)
                    convs = net
                else:
                    convs = BatchNormalization(axis=channel_axis)(net)
                    convs = Activation("relu")(convs)
                convs = Convolution2D(n_bottleneck_plane, nb_col=v[0], nb_row=v[1],
                                     subsample=v[2],
                                     border_mode=v[3],
                                     init=weight_init,
                                     W_regularizer=l2(weight_decay),
                                     bias=use_bias)(convs)
            else:
                convs = BatchNormalization(axis=channel_axis)(convs)
                convs = Activation("relu")(convs)
                if dropout_probability > 0:
                   convs = Dropout(dropout_probability)(convs)
                convs = Convolution2D(n_bottleneck_plane, nb_col=v[0], nb_row=v[1],
                                     subsample=v[2],
                                     border_mode=v[3],
                                     init=weight_init,
                                     W_regularizer=l2(weight_decay),
                                     bias=use_bias)(convs)

        # Shortcut Conntection: identity function or 1x1 convolutional
        #  (depends on difference between input & output shape - this
        #   corresponds to whether we are using the first block in each
        #   group; see _layer() ).
        if n_input_plane != n_output_plane:
            shortcut = Convolution2D(n_output_plane, nb_col=1, nb_row=1,
                                     subsample=stride,
                                     border_mode="same",
                                     init=weight_init,
                                     W_regularizer=l2(weight_decay),
                                     bias=use_bias)(net)
        else:
            shortcut = net

        return merge([convs, shortcut], mode="sum")
    
    return f


# "Stacking Residual Units on the same stage"
def _layer(block, n_input_plane, n_output_plane, count, stride):
    def f(net):
        net = block(n_input_plane, n_output_plane, stride)(net)
        for i in range(2,int(count+1)):
            net = block(n_output_plane, n_output_plane, stride=(1,1))(net)
        return net
    
    return f

def model_fn(labels_dim):
  """Create a Keras Sequential model with layers."""

    assert((depth - 4) % 6 == 0)
    n = (depth - 4) / 6
    
    inputs = Input(shape=input_shape)

    n_stages=[16, 16*k, 32*k, 64*k]

    conv1 = Convolution2D(nb_filter=n_stages[0], nb_row=3, nb_col=3, 
                          subsample=(1, 1),
                          border_mode="same",
                          init=weight_init,
                          W_regularizer=l2(weight_decay),
                          bias=use_bias)(inputs) # "One conv at the beginning (spatial size: 32x32)"

    # Add wide residual blocks
    block_fn = _wide_basic
    conv2 = _layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1,1))(conv1)# "Stage 1 (spatial size: 32x32)"
    conv3 = _layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2,2))(conv2)# "Stage 2 (spatial size: 16x16)"
    conv4 = _layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2,2))(conv3)# "Stage 3 (spatial size: 8x8)"

    batch_norm = BatchNormalization(axis=channel_axis)(conv4)
    relu = Activation("relu")(batch_norm)
                                            
    # Classifier block
    pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), border_mode="same")(relu)
    flatten = Flatten()(pool)
    predictions = Dense(output_dim=nb_classes, init=weight_init, bias=use_bias,
                        W_regularizer=l2(weight_decay), activation="softmax")(flatten)

    model = Model(input=inputs, output=predictions)

  compile_model(model)
  return model

def compile_model(model):
  model.compile(loss = keras.losses.categorical_crossentropy,
                optimizer = sgd,
                metrics = ['accuracy'])
  return model


#Type conversion
def read_train_data():
  #data=np.load("data/trainData.npz") #read from localcomputer
  start_time = time.time()
  print ("Start Read Train Data")
  data = np.load(tf.gfile.Open("gs://retinopathy/data/trainData.npz")) #read from the bucket
  print("Train data read --- %s seconds ---" % (time.time() - start_time))
  X_train=data["X_train"]
  X_train=Xtrain.astype('float32')
  Y_train=data["Y_train"]
  #lower case y 
  y_train=np_utils.to_categorical(Y_train, 5)
  print ("Training - Total examples per class",np.sum(Y_train, axis=0))
  return [X_train,Y_train]

def read_test_data():
  #data=np.load("data/testData.npz") #read from localcomputer
  start_time = time.time()
  print ("Start Read Test Data")
  data = np.load(tf.gfile.Open("gs://retinopathy/data/testData.npz")) #read from the bucket
  print("Test data read --- %s seconds ---" % (time.time() - start_time))
  X_test=data["X_test"]
  X_test=X_test.astype('float32')
  Y_test=data["Y_test"]
  #lower case y
  y_test=np_utils.to_categorical(Y_train, 5)
  print ("Testing - Total examples per class",np.sum(Y_test, axis=0))
  return [X_test,Y_test]