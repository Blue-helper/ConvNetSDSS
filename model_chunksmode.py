#!/usr/bin/env python
# -*- coding: utf-8 -*-
from trainmodel_chunsmode import *
from keras.models import Sequential
from keras.layers import MaxoutDense, Convolution2D, pooling, MaxPooling2D, Flatten, Dense, Dropout, Activation
import scipy as sp
import numpy as np

##### CNN model

model = Sequential()
model.add(Convolution2D(16,6,6,border_mode='same', activation='relu',input_shape=(1,69,69)))
model.add(pooling.MaxPooling2D(pool_size=(2,2),border_mode='same')) #32x32

model.add(Convolution2D(32,5,5,border_mode='same', activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2,2),border_mode='same'))# 14x14

model.add(Convolution2D(64,3,3,border_mode='same', activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2,2),border_mode='same'))#6x6

model.add(Convolution2D(128,3,3,border_mode='same', activation='relu'))#4x4
#model.add(MaxPooling2D(pool_size=(2,2),border_mode='same')) #2x2x256=1024
model.add(Flatten())

#model.add(MaxoutDense(2048))
model.add(Dense(2048,activation = 'relu'))

model.add(Dropout(0.5))
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(3))
model.compile(loss='mean_squared_error', optimizer='sgd')

##### Load data iteration data
data = Data_ConvNetSDSS()
batch_size = 500

for i in range(int(60000/batch_size)):
    x_chunk, y_chunk = data.train_chunks(batch_size)
    model.train_on_batch (x_chunk, y_chunk)

x_val, y_val = data.eval_data()
x_predict, y_predict = data.predict_data()

score = model.evaluate(x_eval, y_eval, batch_size=32)
proba = model.predict_proba(x_predict,batch_size=32)
