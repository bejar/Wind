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
__author__ = 'bejar'


lsites = ["12-6100-12","12-6101-12","12-6102-12","12-6103-12","12-6104-12","12-6105-12","12-6106-12","12-6107-12","12-6108-12"]
wind = {}
for d in lsites:
    print(d)
    if os.path.exists(wind_data_path + f'/{d}.hdf5'):
        hf = h5py.File(wind_data_path + f'/{d}.hdf5', 'r')
        wind[d] = hf[f'wf/Raw'][()]
    else:
        wind[d] = np.load(wind_data_path + f"/{d}.npy")


for d1 in wind:
    for d2 in wind:
        print (d1,d2, np.corrcoef(wind[d1][:,0], wind[d2][:,0])[0,1])