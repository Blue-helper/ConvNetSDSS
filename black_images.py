from __future__ import division
import wget as wg
import pandas as pd
import numpy as np
import collections as cll
import os
from astropy.io import fits
import glob

def black_images():

    files =  glob.glob("./fits/*.fits")
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
