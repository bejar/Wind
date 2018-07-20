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
from Wind.Config.Paths import wind_data_path, wind_path, wind_NREL_data_path
import argparse
from time import strftime


__author__ = 'bejar'


def generate_time_vars(dfile):
    """
    Generates the time variables for one file (the rest are the same)

    :return:
    """
    nc_fid = Dataset(wind_NREL_data_path + "/%s.nc" % dfile, 'r')
    print ( "Read %s" % strftime('%Y-%m-%d %H:%M:%S'))
    nint = nc_fid.dimensions['time'].size
    stime = nc_fid.getncattr('start_time')
    samp = nc_fid.getncattr('sample_period')
    hour = np.array(
        [t.tm_hour * 60 + t.tm_min for t in [time.gmtime(stime + (i * samp)) for i in range(0, nint, step)]])
    month = np.array([t.tm_mon for t in [time.gmtime(stime + (i * samp)) for i in range(0, nint, step)]])
    return hour, month

def generate_data(dfile, vars, step, mode='average', hour=None, month=None):
    """

    :param dfile:
    :param vars:
    :param step:
    :param mode: Mode for multi-step data generation (>1):
        'average' all the points in a step
        'split' the data steps in separated files
    :return:
    """
    nc_fid = Dataset(wind_NREL_data_path + "/%s.nc" % dfile, 'r')

    if step == 1: # The original data
        ldata = []
        for v in vars:
            data = np.nan_to_num(np.array(nc_fid.variables[v]), copy=False)
            ldata.append(data)
        ldata.append(hour)
        ldata.append(month)

        data_stack = np.stack(ldata, axis=1)
        print(data_stack.shape)
        np.save(wind_data_path + '/%s-%02d.npy' % (wf.replace('/', '-'), step), data_stack)
    elif mode == 'average': # Average step points
        ldata = []
        for v in vars:
            data = np.nan_to_num(np.array(nc_fid.variables[v]), copy=False)
            end = data.shape[0]
            length = int(end / step)
            # tmp = data.reshape((length, step)).sum(axis=1)/float(step)

            data_aver = np.zeros(length)
            for i in range(step):
                data_aver += data[i::step]
            data_aver /= float(step)

            ldata.append(data_aver)
        ldata.append(hour)
        ldata.append(month)

        data_stack = np.stack(ldata, axis=1)
        print(data_stack.shape)

        np.save(wind_data_path + '/%s-%02d.npy' % (wf.replace('/', '-'), step), data_stack)

    elif mode == 'split': # split in n step files
        for i in range(step):
            ldata = []
            for v in vars:
                data = np.nan_to_num(np.array(nc_fid.variables[v]), copy=False)
                ldata.append(data[i::step])
            ldata.append(hour)
            ldata.append(month)

            data_stack = np.stack(ldata, axis=1)
            print(data_stack.shape)
            np.save(wind_data_path + '/%s-%02d-%02d.npy' % (wf.replace('/', '-'), step, i+1), data_stack)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--isec', type=int, help='Sites Section')
    parser.add_argument('--fsec', type=int, help='Initial Site')
    parser.add_argument('--isite', type=int, help='Final Site')
    parser.add_argument('--step', default=12, type=int, help='Grouping step')
    parser.add_argument('--mode', default='average', choices=['average', 'split'], help='Grouping mode')
    args = parser.parse_args()

    # Grupos de 5 minutos
    step = args.step

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

    hour, month = generate_time_vars(str(args.isec) +'/' + str(args.isite))

    for c, site in enumerate(range(args.isec, args.fsec+1)):
        wfiles = [str(site)+'/' + str(i) for i in range(args.isite+(c*500), args.isite+500+(c*500)) ]

        for wf in wfiles:
            print("Processing %s" % wf)
            generate_data(wf, vars, step, mode=args.mode, hour=hour, month=month)
