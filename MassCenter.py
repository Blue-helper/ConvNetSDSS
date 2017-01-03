from __future__ import division
import wget as wg
import pandas as pd
import numpy as np
import collections as cll
import os
# Version para 1000 galaxias aprox, debe ser generalizada

from astropy.io import fits
# objid, run, camcol, field
hdulist = fits.open('fits/new'+str(11)+'fpAtlas-00'+str(3900)+'-'+str(4)+'-0'+str(627)+'.fits')
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
print sumMRi/sumM, sumMRj/sumM
maxim = np.amax(data)
center = np.where(data ==maxim)
print center
