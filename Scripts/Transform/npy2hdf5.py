"""
.. module:: npy2hdf5

npy2hdf5
******

:Description: npy2hdf5

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 

:Date:  12/06/2021
"""
from numpy import load
import h5py
from Wind.Config.Paths import wind_data_path, wind_data_path_a, wind_path, wind_NREL_data_path
import argparse
from time import strftime
from tqdm import tqdm


__author__ = 'bejar'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--isec', type=int, default=190, help='Sites Section')
    parser.add_argument('--fsec', type=int, default=190, help='Initial Site')
    args = parser.parse_args()

    for site in tqdm(range(args.isec, args.fsec + 1), desc='Section'):
        wfiles = [f"{site}-{i}-12" for i in range(site * 500, (site+1) * 500)]
        for wf in tqdm(wfiles, desc='Site'):
            a = load(f'{wind_data_path}/{wf}.npy')
            f = h5py.File('f{wind_data_path}/{wf}.hdf5', 'w')
            dgroup = f.create_group('wf')
            dgroup.create_dataset('Raw', a.shape, dtype='f', data=a, compression='gzip')
            f.flush()
            f.close()

