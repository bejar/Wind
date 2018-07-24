"""
.. module:: MapSites

MapSites
*************

:Description: MapSites

    

:Authors: bejar
    

:Version: 

:Created on: 15/05/2018 8:46 

"""

from netCDF4 import Dataset
from Wind.Config.Paths import wind_data_path, wind_data_ext, wind_path, wind_NREL_data_path
from Wind.Maps.Util import MapThis
import os
import time
import numpy as np


def explore_files(dir, ds):
    for v in os.listdir(dir + '/' + ds):
        if v[0] in '0123456789':
            yield dir + '/' + ds + '/' + v

def map_all(path):
    lds = [v for v in sorted(os.listdir(path)) if v[0] in '0123456789']
    coords = np.zeros((126692, 2))
    for ds in lds:
        print(ds)
        # lcoords = []
        # lfnames = []
        for f in explore_files(path, ds):
            data = Dataset(f, 'r')
            pos = int(f.split('/')[-1].split('.')[0])
            coords[pos][0]= data.latitude
            coords[pos][1]= data.longitude

            print(pos, [data.latitude, data.longitude])
        # MapThis(lcoords, ds, lfnames)
    np.save(wind_data_path+'/coords.npy',coords)

__author__ = 'bejar'

if __name__ == '__main__':
    map_all(wind_NREL_data_path)