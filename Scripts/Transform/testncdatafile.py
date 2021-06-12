"""
.. module:: testangle

testangle
*************

:Description: testangle

    

:Authors: bejar
    

:Version: 

:Created on: 02/05/2019 10:31 

"""

from __future__ import print_function
from netCDF4 import Dataset
import numpy as np
import time

__author__ = 'bejar'

if __name__ == '__main__':
    wfiles = ['95125']

    vars = ['wind_speed', 'temperature', 'density', 'pressure', 'wind_direction']
    mdata = {}
    for d, wf in enumerate(wfiles):
        print("/home/bejar/storage/Data/Wind/files/%s.nc" % wf)
        nc_fid = Dataset("/home/bejar/storage/Data/Wind/files/%s.nc" % wf, 'r')
        nint = nc_fid.dimensions['time'].size
        print(nint)
        stime = nc_fid.getncattr('start_time')
        samp = nc_fid.getncattr('sample_period')
        # print(np.sin(np.deg2rad(nc_fid.variables['wind_direction'][0:100])))
        for v in vars:
            print(v)
            print(np.max(nc_fid.variables[v]))
            print(np.min(nc_fid.variables[v]))

            # print(nc_fid.variables[v][-1000:])


