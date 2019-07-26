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
import multiprocessing
import json
from time import time, strftime

__author__ = 'bejar'


vars = ['wind_speed', 'temperature', 'density', 'pressure', 'wind_direction']

def saveconfig(site, results, tstamp):
    """
    Saves a config in the database
    :param proxy:
    :return:
    """
    config = {}
    config['experiment'] = 'measures'
    config['site'] = f"{site//500}-{site}"
    config['result'] = results
    config['etime'] = strftime('%Y-%m-%d %H:%M:%S')
    config['_id'] = f'{tstamp}{site:05d}'
    sconf = json.dumps(config)
    fconf = open(wind_res_path + '/measure' + config['_id'] + '.json', 'w')
    fconf.write(sconf + '\n')
    fconf.close()


def compute_values(lsites, windows, tstamp):
    """
    Computes Dataset values
    """

    for s in lsites:
        dmeasures = {}
        for i, v in enumerate(vars):
            dataset = Dataset(config={"datanames": [f"{s//500}-{s}-12"],  "vars": "all"}, data_path=wind_data_path)
            dataset.load_raw_data()
            dmeasures[v] = dataset.compute_measures(i, window=windows)

        saveconfig(s, dmeasures, tstamp)

if __name__ == '__main__':

    #nsites = 126691
    nsites = 100

    windows = {'12h':12, '24h':24, '1w':168, '1m':720, '3m':2190, '6m':4380}
    lsites = range(nsites)
    ncores = 40  # multiprocessing.cpu_count()
    print(ncores)
    # ncores = args.cores
    tstamp = str(int(time() * 10000))
    lparts = []
    for i in range(ncores):
        lparts.append(lsites[i*len(lsites)//ncores:(i+1)*len(lsites)//ncores])

    res = Parallel(n_jobs=ncores)(delayed(compute_values)(sites, windows, tstamp) for sites in lparts)


