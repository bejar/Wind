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
from tqdm import tqdm

__author__ = 'bejar'


def generate_time_vars(dfile):
    """
    Generates the time variables for one file (the rest are the same)

    :return:
    """
    nc_fid = Dataset(f"{wind_NREL_data_path}/{dfile}.nc", 'r')
    print(f"Read {strftime('%Y-%m-%d %H:%M:%S')}")
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
    nc_fid = Dataset(wind_NREL_data_path + f"/{dfile}.nc", 'r')

    if step == 1:  # The original data
        ldata = []
        for v in vars:
            data = np.nan_to_num(np.array(nc_fid.variables[v]), copy=False)
            ldata.append(data)
        ldata.append(hour)
        ldata.append(month)

        data_stack = np.stack(ldata, axis=1)
        np.save(wind_data_path + f"/{wf.replace('/', '-')}-{step:02d}.npy", data_stack)
    elif mode == 'average':  # Average step points
        ldata = []
        for v in vars:
            data = np.nan_to_num(np.array(nc_fid.variables[v]), copy=False)
            end = data.shape[0]
            length = int(end / step)

            data_aver = np.zeros(length)
            for i in range(step):
                data_aver += data[i::step]
            data_aver /= float(step)

            ldata.append(data_aver)
        ldata.append(hour)
        ldata.append(month)

        data_stack = np.stack(ldata, axis=1)

        np.save(wind_data_path + f"/{wf.replace('/', '-')}-{step:02d}.npy", data_stack)

    elif mode == 'split':  # split in n step files
        for i in range(step):
            ldata = []
            for v in vars:
                data = np.nan_to_num(np.array(nc_fid.variables[v]), copy=False)
                ldata.append(data[i::step])
            ldata.append(hour)
            ldata.append(month)

            data_stack = np.stack(ldata, axis=1)
            np.save(wind_data_path + f"/{wf.replace('/', '-')}-{step:02d}-{(i+1):02d}.npy", data_stack)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--isec', type=int, help='Sites Section')
    parser.add_argument('--fsec', type=int, help='Initial Site')
    parser.add_argument('--step', default=12, type=int, help='Grouping step')
    parser.add_argument('--mode', default='average', choices=['average', 'split'], help='Grouping mode')
    args = parser.parse_args()

    # Grupos de step minutos
    step = args.step

    vars = ['wind_speed', 'temperature', 'density', 'pressure', 'wind_direction']

    hour, month = generate_time_vars(f"{args.isec}/{args.isec*500}")

    for site in range(args.isec, args.fsec + 1):
        wfiles = [f"{site}/{i}" for i in range(site * 500, (site+1) * 500)]

        for wf in tqdm(wfiles):
            generate_data(wf, vars, step, mode=args.mode, hour=hour, month=month)
