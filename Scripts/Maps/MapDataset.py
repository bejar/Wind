"""
.. module:: MapDataset

MapDataset
*************

:Description: MapDataset

    

:Authors: bejar
    

:Version: 

:Created on: 21/06/2019 13:09 

"""

import argparse
import numpy as np
import os
from Wind.Config.Paths import wind_data_path
from Wind.Data import Dataset
from Wind.Util.Maps import  create_plot
import pandas as pd
from tqdm import tqdm

__author__ = 'bejar'



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=float, default=0.001, help='Sampling of the sites')
    args = parser.parse_args()
    if os.path.isfile(f'{wind_data_path}/Coords.npy'):
        coords = np.load(wind_data_path + '/Coords.npy')
    else:
        raise NameError('No coordinates file found')

    nsites = 126691
    lsites = np.random.choice(range(nsites), int(args.sample*nsites), replace=False)

    dres = {'Lat':[], 'Lon':[], 'Val':[], 'Site':[]}
    for s in tqdm(lsites):
        print(s)
        dataset = Dataset(config={"datanames": [f"{s//500}-{s}-12"],  "vars": [0]}, data_path=wind_data_path)
        dataset.load_raw_data()
        res = dataset.compute_measures(window=[12])
        dres['Lat'].append(coords[s][1])
        dres['Lon'].append(coords[s][0])
        dres['Val'].append(res['specent'])
        dres['Site'].append(f"{s//500}-{s}")

    df = pd.DataFrame(dres)
    create_plot(df, 'specent')
