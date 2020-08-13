"""
.. module:: DataMeasures

DataMeasures
*************

:Description: DataMeasures

    

:Authors: bejar
    

:Version: 

:Created on: 22/07/2019 9:11 

"""

import numpy as np
import os
from Wind.Config.Paths import wind_data_path, wind_res_path
from Wind.Data import Dataset
from joblib import Parallel, delayed
import json
from time import time, strftime
import argparse

__author__ = 'bejar'


vars = ['wind_speed', 'temperature', 'density', 'pressure', 'wind_direction_cos', 'wind_direction_sin']

def saveconfig(site, results, tstamp):
    """
    Saves a config in the database
    :param proxy:
    :return:
    """
    config = {'experiment': 'measures', 'site': f"{site // 500}-{site}", 'result': results,
              'etime': strftime('%Y-%m-%d %H:%M:%S'), '_id': f'{tstamp}{site:05d}'}
    sconf = json.dumps(config)
    fconf = open(wind_res_path + '/measure' + config['_id'] + '.json', 'w')
    fconf.write(sconf + '\n')
    fconf.close()


def compute_values(lsites, windows, tstamp, measures):
    """
    Computes Dataset values
    """

    for s in lsites:
        print(s, flush=True)
        dmeasures = {}
        for i, v in enumerate(vars):
            dataset = Dataset(config={"datanames": [f"{s//500}-{s}-12"],  "vars": "all"}, data_path=wind_data_path)
            dataset.load_raw_data()
            dmeasures[v] = dataset.compute_decomposition(i, window=windows) if measures == 'stl' else dataset.compute_measures(i, window=windows)

        saveconfig(s, dmeasures, tstamp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--isite', type=int, default=0,  help='Initial Site')
    parser.add_argument('--nsites', type=int, default=10000,  help='Number of sites')
    parser.add_argument('--ncores', type=int, default=40, help='Experiment ID')
    parser.add_argument('--measures', type=str, default='stl', help='type of measures')

    args = parser.parse_args()

    lim_sites = 126692
    # nsites = 100
    if args.isite > lim_sites:
        raise NameError('Initial site out of bounds')

    windows = {'12h':12, '24h':24, '1w':168, '1m':720, '3m':2190, '6m':4380}
    if (args.isite+args.nsites) <= lim_sites:
        lsites = range(args.isite, args.nsites+args.nsites)
    else:
        lsites = range(args.isite, lim_sites)
    ncores = args.ncores  # multiprocessing.cpu_count()
    tstamp = str(int(time() * 10000))
    lparts = []
    for i in range(ncores):
        lparts.append(lsites[i*len(lsites)//ncores:(i+1)*len(lsites)//ncores])

    res = Parallel(n_jobs=ncores)(delayed(compute_values)(sites, windows, tstamp, args.measures) for sites in lparts)


