from __future__ import division
import wget as wg
import pandas as pd
import numpy as np
import collections as cll
import sys
import os
'''
python fetcher.py numberofobject
if numberofobject = 0 then all images will be downloaded

'''
GZ2 = pd.read_csv('GalaxyZoo2.csv')

run = np.array(GZ2['run'].values, dtype=str)
rerun = np.array(GZ2['rerun'].values, dtype=str)
camcol = np.array(GZ2['camcol'].values, dtype=str)
field = np.array(GZ2['field'].values, dtype=str)

aux_list = []


for i in range(len(GZ2)):
    #https://data.sdss.org/sas/dr13/eboss/photo/redux/157/1933/objcs/2/fpAtlas-001933-2-0011.fit
    k = 'https://data.sdss.org/sas/dr13/eboss/photo/redux/'+rerun[i]+'/'+run[i]+'/objcs/'+camcol[i]+'/fpAtlas-'+run[i].zfill(6)+'-'+camcol[i]+'-'+field[i].zfill(4)+'.fit'
    aux_list.append(k)
    print '\r '+ str(i),

contador = cll.Counter(aux_list)

todownload = int(sys.argv[1])
if todownload==0:
    todownload = len(contador)
    atleastarg = contador.most_common(int(todownload))
else:
    atleastarg = contador.most_common(int(todownload/5))

c = 0 #download counter

for i in range(len(atleastarg)):
    c+= atleastarg[i][1]
    wg.download(atleastarg[i][0])
    if c >= todownload:
        print i
        break
if not os.path.exists(fit):
    os.makedirs(fit)
os.system('mv *.fit ./fit/')
