"""
.. module:: TransformData

TransformData
*************

:Description: TransformData

    

:Authors: bejar
    

:Version: 

:Created on: 12/04/2018 10:21 

"""

from __future__ import print_function
from netCDF4 import Dataset
import numpy as np
import time

__author__ = 'bejar'


def generate_data(dfile, vars, step):
    """

    :param dfile:
    :param vars:
    :param step:
    :return:
    """

    nc_fid = Dataset("/home/bejar/storage/Data/Wind/files/%s.nc" % dfile, 'r')
    nint = nc_fid.dimensions['time'].size
    stime = nc_fid.getncattr('start_time')
    samp = nc_fid.getncattr('sample_period')
    hour = np.array(
        [t.tm_hour * 60 + t.tm_min for t in [time.gmtime(stime + (i * samp)) for i in range(0, nint, step)]])
    month = np.array([t.tm_mon for t in [time.gmtime(stime + (i * samp)) for i in range(0, nint, step)]])

    ldata = []
    for v in vars:
        data = nc_fid.variables[v]

        if step == 1:
            ldata.append(data)
        else:
            end = data.shape[0]
            length = int(end / step)
            print(length)
            data_aver = np.zeros(length)

            for i in range(0, end, step):
                data_aver[i / step] = np.sum(data[i: i + step]) / step

            ldata.append(data_aver)
    ldata.append(hour)
    ldata.append(month)

    data_stack = np.stack(ldata, axis=1)
    print(data_stack.shape)
    np.save('/home/bejar/%s-%d.npy' % (wf.replace('/', '-'), step), data_stack)


if __name__ == '__main__':

    # Grupos de 5 minutos
    step = 1

    # wfiles = ['90/45142', '90/45143',
    #           '90/45229','90/45230']
    # wfiles = ['1/741', '1/742', '1/743',
    #           '1/703', '1/704', '1/705',
    #           '1/668', '1/669', '1/670']
    # wfiles = ['11/5883', '11/5884', '11/5885', '11/5886',
    #           '11/5836', '11/5837', '11/5838', '11/5839',
    #           '11/5793', '11/5794', '11/5795', '11/5796',
    #           '11/5752', '11/5753', '11/5754', '11/5755']
    vars = ['wind_speed', 'density', 'pressure', 'wind_direction']
    wfiles = ['11/5794']

    for wf in wfiles:
        print("Processing %s" % wf)
        generate_data(wf, vars, step)
