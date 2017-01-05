#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
import numpy as np
import glob
from astropy.io import fits
from skimage import transform
import os

def downsampler():
    files =  glob.glob("./kerasfits/*.fits")

    for pathfile in files:
        hdulist = fits.open(pathfile)
        data = hdulist[0].data

        downsample_matrix = transform.downscale_local_mean(data,(69,69))
        hdulist[0].data = downsample_matrix
        if not os.path.exists('kerasfits_ds'):
            os.makedirs('kerasfits_ds')
        hdulist.writeto('./kerasfits_ds/'+pathfile.split('/')[-1])
