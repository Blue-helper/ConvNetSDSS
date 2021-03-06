from __future__ import division
import wget as wg
import pandas as pd
import numpy as np
import collections as cll
import os
from astropy.io import fits
import glob

def image_creator():

    files =  glob.glob("./fits/*.fits")


    # objid, run, camcol, field
    blackmatrix = np.ones((207,207))*1000
    for pathfile in files:
        hdulist = fits.open(pathfile)
        data = hdulist[0].data

        # Centro de masa
        sumMRi = 0
        sumMRj = 0
        sumM = 0
        for i in range(len(data)):
            for j in range(len(data[0])):
                sumMRi += (data[i,j]-1000)*i
                sumMRj += (data[i,j]-1000)*j
                sumM += (data[i,j]-1000)
        #funciona
        if sumM==0:
            print pathfile
            continue
        else:
            centerx, centery = [int(sumMRi/sumM), int(sumMRj/sumM)]

        for i in range(207):
            for j in range(207): #hacerlo al reves
                if i+centerx-103>=0 and j+centery-103>=0: #si i o j es mas grande que len data no hacer nada
                    try:
                        blackmatrix[i,j] = data[i+centerx-103,j+centery-103]
                    except:
                        continue
        hdulist[0].data = blackmatrix
        if not os.path.exists('kerasfits'):
            os.makedirs('kerasfits')
        hdulist.writeto('./kerasfits/'+pathfile.split('/')[-1])
