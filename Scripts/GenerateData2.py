"""
.. module:: GenerateData

GenerateData
*************

:Description: GenerateData

    Genera un fichero de datos con los sites y variables indicadas con una resolucion indicada por step

:Authors: bejar
    

:Version: 

:Created on: 07/02/2018 10:10 

"""

from __future__ import print_function
from netCDF4 import Dataset
import numpy as np
import time

__author__ = 'bejar'


if __name__ == '__main__':

    # Grupos de 5 minutos
    step = 12

    wfiles = ['90/45142', '90/45143','90/45229','90/45230']
    vars = ['wind_speed', 'density', 'pressure', 'wind_direction']
    mdata = {}
    for d, wf in enumerate(wfiles):
        print("/home/bejar/storage/Data/Wind/files/%s.nc" % wf)
        nc_fid = Dataset("/home/bejar/storage/Data/Wind/files/%s.nc" % wf, 'r')
        nint = nc_fid.dimensions['time'].size
        stime = nc_fid.getncattr('start_time')
        samp = nc_fid.getncattr('sample_period')
        hour = np.array([t.tm_hour * 60 + t.tm_min for t in [time.gmtime(stime + (i * samp)) for i in range(0, nint, step)]])
        month = np.array([t.tm_mon for t in [time.gmtime(stime + (i * samp)) for i in range(0, nint, step)]])

        ldata = []
        for v in vars:
            data = nc_fid.variables[v]
            print(data.shape)

            end = data.shape[0]
            length = int(end/step)
            print(length)
            data_aver = np.zeros(length)

            for i in range(0, end, step):
                data_aver[i / step] = np.sum(data[i: i + step]) / step

            ldata.append((data_aver))
        ldata.append(hour)
        ldata.append(month)

        data_stack = np.stack(ldata, axis=1)
        print(data_stack.shape)
        np.save('/home/bejar/%s.npy' % wf.replace('/', '-'), data_stack)

