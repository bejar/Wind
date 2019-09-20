"""
.. module:: TransformData

TransformData
*************

:Description: TransformData

 Transforms Data for a generated sample of sites
    

:Authors: bejar
    

:Version: 

:Created on: 12/04/2018 10:21 

"""

from netCDF4 import Dataset
import numpy as np
import time
from Wind.Config.Paths import wind_data_path, wind_data_path_a, wind_path, wind_NREL_data_path
import argparse
from time import strftime
from tqdm import tqdm
from Wind.Private.DBConfig import mongoconnection, mongolocaltest
from pymongo import MongoClient

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
            if v != 'wind_direction':
                ldata.append(data)
            else:
                data = np.deg2rad(data)
                datasin = np.sin(data)
                datacos = np.cos(data)
                ldata.append(datasin)
                ldata.append(datacos)
        ldata.append(hour)
        ldata.append(month)

        data_stack = np.stack(ldata, axis=1)
        np.save(wind_data_path + f"/{wf.replace('/', '-')}-{step:02d}.npy", data_stack)
    elif mode == 'average':  # Average step points
        ldata = []
        for v in vars:
            data = np.nan_to_num(np.array(nc_fid.variables[v]), copy=False)
            if v != 'wind_direction':
                end = data.shape[0]
                length = int(end / step)

                data_aver = np.zeros(length)
                for i in range(step):
                    data_aver += data[i::step]
                data_aver /= float(step)
                ldata.append(data_aver)
            else:  # Angle data is averaged differently
                data = np.deg2rad(data)
                datasin = np.sin(data)
                datacos = np.cos(data)
                end = data.shape[0]
                length = int(end / step)

                datasin_aver = np.zeros(length)
                datacos_aver = np.zeros(length)
                for i in range(step):
                    datasin_aver += datasin[i::step]
                    datacos_aver += datacos[i::step]
                    
                datacos_aver /= float(step)
                datasin_aver /= float(step)
        
                ldata.append(datasin_aver)
                ldata.append(datacos_aver)
        ldata.append(hour)
        ldata.append(month)

        data_stack = np.stack(ldata, axis=1)

        np.save(wind_data_path_a + f"/{wf.replace('/', '-')}-{step:02d}.npy", data_stack)

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
    elif mode == 'minmax':  # Average, max and min step points
        ldata = []
        # Average Values
        for v in vars:
            data = np.nan_to_num(np.array(nc_fid.variables[v]), copy=False)
            ldata.append(np.mean(data.reshape((-1, step)), axis=1))
        # Min values
        for v in vars:
            data = np.nan_to_num(np.array(nc_fid.variables[v]), copy=False)
            ldata.append(np.min(data.reshape((-1, step)), axis=1))
        # Max Values
        for v in vars:
            data = np.nan_to_num(np.array(nc_fid.variables[v]), copy=False)
            ldata.append(np.max(data.reshape((-1, step)), axis=1))
        ldata.append(hour)
        ldata.append(month)

        data_stack = np.stack(ldata, axis=1)

        np.save(wind_data_path + f"/{wf.replace('/', '-')}-{step:02d}-amm.npy", data_stack)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str, help='Initial Site')
    parser.add_argument('--step', default=12, type=int, help='Grouping step')
    parser.add_argument('--mode', default='average', choices=['average', 'split'], help='Grouping mode')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')
    args = parser.parse_args()

    mongoconnection = mongolocaltest
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.passwd is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    # Check if sample exists
    sample = col.find_one({'sample': 'init', 'sname': args.sample})

    if sample is not None:
        ssites =  sample['sites']



    # Grupos de step minutos
    step = args.step

    vars = ['wind_speed', 'temperature', 'density', 'pressure', 'wind_direction']
    nsite = ssites[0].split('-')
    hour, month = generate_time_vars(f"{nsite[0]}/{nsite[1]}")

    for site in tqdm(ssites):
        nsite = site.split('-')
        wf = f"{nsite[0]}/{nsite[1]}"
        generate_data(wf, vars, step, mode=args.mode, hour=hour, month=month)
