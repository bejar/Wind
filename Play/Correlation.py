"""
.. module:: Correlation

Correlation
*************

:Description: Correlation

    

:Authors: bejar
    

:Version: 

:Created on: 12/07/2021 9:11 

"""
import os
import numpy as np
import h5py
from Wind.Config.Paths import wind_data_path
from Wind.Spatial.Util import SitesCoords
__author__ = 'bejar'


lsites = ["11-5795-12", "11-5791-12","11-5792-12","11-5793-12","11-5794-12","11-5796-12","11-5797-12",
          "11-5798-12","11-5799-12",]
wind = {}
for d in lsites:
    print(d)
    if os.path.exists(wind_data_path + f'/{d}.hdf5'):
        hf = h5py.File(wind_data_path + f'/{d}.hdf5', 'r')
        wind[d] = hf[f'wf/Raw'][()]
    else:
        wind[d] = np.load(wind_data_path + f"/{d}.npy")

sc = SitesCoords()

for d1 in wind:
    for d2 in wind:
        s1 = d1.split('-')
        s2 = d2.split('-')
        print (d1,d2, np.corrcoef(wind[d1][:,0], wind[d2][:,0])[0,1], sc.get_coords(int(s1[1])) ,sc.get_coords(int(s2[1]))  )