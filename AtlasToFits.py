from __future__ import division
import wget as wg
import pandas as pd
import numpy as np
import collections as cll
import os
# Version para 1000 galaxias aprox, debe ser generalizada

GZ2 = pd.read_csv('GalaxyZoo2.csv')

Objid = np.array(GZ2['obj'].values, dtype=str)
run = np.array(GZ2['run'].values, dtype=str)
rerun = np.array(GZ2['rerun'].values, dtype=str)
camcol = np.array(GZ2['camcol'].values, dtype=str)
field = np.array(GZ2['field'].values, dtype=str)

aux_list = []

for i in range(len(GZ2)):
    k = 'https://data.sdss.org/sas/dr13/eboss/photo/redux/'+rerun[i]+'/'+run[i]+'/objcs/'+camcol[i]+'/fpAtlas-'+run[i].zfill(6)+'-'+camcol[i]+'-'+field[i].zfill(4)+'.fit'
    aux_list.append(k)
    print '\r '+ str(i),

aux_list = np.array(aux_list)
contador = cll.Counter(aux_list)

atleast1000 = contador.most_common(100)

c = 0
MyIDS =[]

for i in range(len(atleast1000)):
    c+= atleast1000[i][1]
    MyIDS.append([atleast1000[i][0], Objid[np.where(aux_list == atleast1000[i][0])[0]]] )
    if c >= 1000:
        print i
        break


for val in MyIDS:
    for obj in val[1]:
        # Put here ur atlas path
        command ='../readAtlasImages-v5_4_11/read_atlas_image -c 2 fit/'+val[0].split('/')[-1]+' '+obj+"  new"+obj+val[0].split('/')[-1]+'s'
        os.system(command)
