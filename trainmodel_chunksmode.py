#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import scipy as sp
import numpy as np
import glob
from astropy.io import fits
import gc

##### train data and labels(probabilities)
#labels
Class Data_ConvNetSDSS():
    def __init__(self)
    GZ2 = pd.read_csv('GalaxyZoo2.csv')
    r1 = GZ2["t01_smooth_or_features_a01_smooth_debiased"]
    r2 = GZ2["t01_smooth_or_features_a02_features_or_disk_debiased"]
    r3 = GZ2["t01_smooth_or_features_a03_star_or_artifact_debiased"]
    self.actual_used_data = 0
    #Train images
    files =  glob.glob("./kerasfits_ds/*.fits")
    self.x_train = []
    self.y_train = []
    self.x_eval = []
    self.y_eval = []
    self.x_predict = []
    self.y_predict = []
    size_evaluate_data = 1000
    size_predict_data = 1000
    size_train_data = len(files)-size_predict_data-size_evaluate_data
    dummy_counter = 0 # counter of trainning data
    for pathfile in files:
        hdulist = fits.open(pathfile)
        data = hdulist[0].data.copy() 
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
            self.x_train.append(data)
            self.y_train.append([r1[d],r2[d],r3[d]])
        elif size_train_data<dummy_counter and dummy_counter - size_train_data <1000:
            self.x_eval.append(data)
            self.y_eval.append([r1[d],r2[d],r3[d]])
        else:
            self.x_predict.append(data)
            self.y_predict.append([r1[d],r2[d],r3[d]])
        del hdulist[0].data
        del data
        hdulist.close()
        gc.collect()
        dummy_counter += 1
        print '\r '+ str(dummy_counter*100/len(files))+'%',

    def train_chunks(batch_size):
        self.actual_used_data +=1
        if self.actual_used_data*batch_size > len(self.x_train):
            print "not enough data"
        else :
            x_chunk = sp.asarray(self.x_train[(self.actual_used_data-1)32:32*(self.actual_used_data)])
            x_chunk = x_train.reshape((self.x_train.shape[0],1,self.x_train.shape[1],self.x_train.shape[2]))
            return  x_chunk, sp.asarray(self.y_train)

    def eval_data():

        x_eval = sp.asarray(x_eval)
        x_eval = x_eval.reshape((x_eval.shape[0],1,x_eval.shape[1],x_eval.shape[2]))
        y_eval = sp.asarray(y_eval)
        return x_eval, y_eval

    def predict_data():

        x_predict = sp.asarray(x_predict)
        x_predict = x_predict.reshape((x_predict.shape[0],1,x_predict.shape[1],x_predict.shape[2]))
        y_predict = sp.asarray(y_predict)
