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
from Wind.Config import wind_data_path, wind_data_ext, wind_path
from Wind.Maps.Util import MapThis
import os
import time


def explore_files(dir, ds):
    for v in os.listdir(dir + '/' + ds):
        if v[0] in '0123456789':
            yield dir + '/' + ds + '/' + v

def map_all(path):
    lds = [v for v in sorted(os.listdir(path)) if v[0] in '0123456789']
    print(lds)
    for ds in lds:
        print(ds)
        lcoords = []
        lfnames = []
        for f in explore_files(path, ds):
            print(f)
            data = Dataset(f, 'r')
            lcoords.append([data.latitude, data.longitude])
            lfnames.append(f)

        MapThis(lcoords, ds, lfnames)

__author__ = 'bejar'

if __name__ == '__main__':
    map_all(wind_path+'/files')