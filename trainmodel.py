#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import MaxoutDense, Convolution2D, pooling, MaxPooling2D, Flatten, Dense, Dropout, Activation
import pandas as pd
import scipy as sp
import numpy as np
import glob
from astropy.io import fits

#import tensorflow as tf
#from keras import backend as K
#K.set_session(tf.Session())

##### train data and labels(probabilities)
#labels
GZ2 = pd.read_csv('GalaxyZoo2.csv')
r1 = GZ2["t01_smooth_or_features_a01_smooth_debiased"]
r2 = GZ2["t01_smooth_or_features_a02_features_or_disk_debiased"]
r3 = GZ2["t01_smooth_or_features_a03_star_or_artifact_debiased"]
prob = [1]*len(r1)
for i in range(len(r1)):
    maxprob = max(r1[i],r2[i],r3[i])
    if r1[i]==maxprob :
        prob[i] = 1
    elif r2[i]==maxprob:
        prob[i] = 2
    elif r3[i]==maxprob:
        prob[i] = 3

#Train images
files =  glob.glob("./kerasfits/*.fits")
x_train = []
y_train = []
for pathfile in files:
    hdulist = fits.open(pathfile)
    data = hdulist[0].data
    mindata, maxdata = np.min(data), np.max(data)
    for i in range(207):
        for j in range(207):
            data[i,j] = (data[i,j]-mindata)/(maxdata-mindata)

    x_train.append(data)
    run = int(pathfile.split('/')[-1][-18:-12])
    camcol = int(pathfile.split('/')[-1][-11:-10])
    field = int(pathfile.split('/')[-1][-9:-5])
    objid = int(pathfile.split('/')[-1].replace("new","")[0:-26])
    d = list(set(sp.where(GZ2["run"]==run)[0]).intersection(sp.where(GZ2["camcol"]==camcol)[0],sp.where(GZ2["field"]==field)[0], sp.where(GZ2["obj"]==objid)[0] ))[0]
    y_train.append([r1[d],r2[d],r3[d]])
x_train = sp.asarray(x_train)
x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1],x_train.shape[2]))
y_train = sp.asarray(y_train)

##### CNN model

model = Sequential()
model.add(Convolution2D(16,10,10,border_mode='same', activation='relu',input_shape=(1,207,207)))
model.add(pooling.MaxPooling2D(pool_size=(2,2),border_mode='same')) #98x98

model.add(Convolution2D(32,9,9,border_mode='same', activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2,2),border_mode='same'))# 45x45

model.add(Convolution2D(64,6,6,border_mode='same', activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2,2),border_mode='same'))#20x20

model.add(Convolution2D(128,5,5,border_mode='same', activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2,2),border_mode='same')) #8x8

model.add(Convolution2D(256,3,3,border_mode='same', activation='relu'))#6x6


model.add(Convolution2D(256,3,3,border_mode='same', activation='relu'))#4x4
model.add(MaxPooling2D(pool_size=(2,2),border_mode='same')) #2x2x256=1024
model.add(Flatten())

#model.add(MaxoutDense(2048))
model.add(Dense(256))
model.add(Dense(3)) #?
model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit(x_train, y_train, batch_size=32, nb_epoch=2)
