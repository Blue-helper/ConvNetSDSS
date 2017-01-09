#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import scipy as sp
import numpy as np
import glob
from astropy.io import fits
import gc
#import tensorflow as tf
#from keras import backend as K
#K.set_session(tf.Session())

##### train data and labels(probabilities)
#labels
GZ2 = pd.read_csv('GalaxyZoo2.csv')
r1 = GZ2["t01_smooth_or_features_a01_smooth_debiased"]
r2 = GZ2["t01_smooth_or_features_a02_features_or_disk_debiased"]
r3 = GZ2["t01_smooth_or_features_a03_star_or_artifact_debiased"]


#Train images
files =  glob.glob("./kerasfits_ds_bup/*.fits")
x_train = []
y_train = []
x_eval = []
y_eval = []
x_predict = []
y_predict = []
size_evaluate_data = 1000
size_predict_data = 1000
size_train_data = len(files)-size_predict_data-size_evaluate_data
dummy_counter = 0 # counter of trainning data
for pathfile in files:
    hdulist = fits.open(pathfile)
    data = hdulist[0].data.copy() #memory error, change list for numpy array, reduce images size
    #image normalization
    mindata, maxdata = np.min(data), np.max(data)
    for i in range(69):
        for j in range(69):
            data[i,j] = (data[i,j]-mindata)/(maxdata-mindata)

    run = int(pathfile.split('/')[-1][-18:-12])
    camcol = int(pathfile.split('/')[-1][-11:-10])
    field = int(pathfile.split('/')[-1][-9:-5])
    objid = int(pathfile.split('/')[-1].replace("new","")[0:-26])
    d = list(set(sp.where(GZ2["run"]==run)[0]).intersection(sp.where(GZ2["camcol"]==camcol)[0],sp.where(GZ2["field"]==field)[0], sp.where(GZ2["obj"]==objid)[0] ))[0]
    if dummy_counter<size_train_data:
        x_train.append(data)
        y_train.append([r1[d],r2[d],r3[d]])
    elif size_train_data<dummy_counter and dummy_counter - size_train_data <1000:
        x_eval.append(data)
        y_eval.append([r1[d],r2[d],r3[d]])
    else:
        x_predict.append(data)
        y_predict.append([r1[d],r2[d],r3[d]])
    del hdulist[0].data
    del data
    hdulist.close()
    gc.collect()
    dummy_counter += 1
    print '\r '+ str(dummy_counter*100/len(files))+'%',
x_train = sp.asarray(x_train)
x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1],x_train.shape[2]))
y_train = sp.asarray(y_train)

x_eval = sp.asarray(x_eval)
x_eval = x_eval.reshape((x_eval.shape[0],1,x_eval.shape[1],x_eval.shape[2]))
y_eval = sp.asarray(y_eval)

x_predict = sp.asarray(x_predict)
x_predict = x_predict.reshape((x_predict.shape[0],1,x_predict.shape[1],x_predict.shape[2]))
y_predict = sp.asarray(y_predict)

np.save('x_train',x_train)
np.save('y_train',y_train)

np.save('x_eval',x_eval)
np.save('y_eval',y_eval)

np.save('x_predict',x_predict)
np.save('y_predict',y_predict)
