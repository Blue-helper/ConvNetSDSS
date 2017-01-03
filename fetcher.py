from __future__ import division
import wget as wg
import pandas as pd
import numpy as np
import collections as cll

GZ2 = pd.read_csv('GalaxyZoo2.csv')

run = np.array(GZ2['run'].values, dtype=str)
rerun = np.array(GZ2['rerun'].values, dtype=str)
camcol = np.array(GZ2['camcol'].values, dtype=str)
field = np.array(GZ2['field'].values, dtype=str)

aux_list = []

#dic_aux = {}
for i in range(len(GZ2)):
    #https://data.sdss.org/sas/dr13/eboss/photo/redux/157/1933/objcs/2/fpAtlas-001933-2-0011.fit
    k = 'https://data.sdss.org/sas/dr13/eboss/photo/redux/'+rerun[i]+'/'+run[i]+'/objcs/'+camcol[i]+'/fpAtlas-'+run[i].zfill(6)+'-'+camcol[i]+'-'+field[i].zfill(4)+'.fit'
    aux_list.append(k)
    print '\r '+ str(i),

contador = cll.Counter(aux_list)

atleast1000 = contador.most_common(100)

c = 0

for i in range(len(atleast1000)):
    c+= atleast1000[i][1]

    wg.download(atleast1000[i][0])
    if c >= 1000:
        print i
        break

    #print atleast1000[i][0]
