"""
.. module:: GenerateExpRangeGeo

GenerateExpRangeGeo
*************

:Description: GenerateExpRangeGeo

    Generates and uploads to the DB configurations using --config configuration
    it begin at files at lat/log --igeo and ends at files ar lat/log --fgeo
    It uses files with suffix --suff

:Authors: bejar
    

:Version: 

:Created on: 07/06/2018 15:45 

"""

import argparse
from time import time

from Wind.Misc import load_config_file
from Wind.Private.DBConfig import mongoconnection, mongolocaltest
from pymongo import MongoClient
from Wind.Config.Paths import wind_data_path
import numpy as np
from tqdm import tqdm

__author__ = 'bejar'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, required=True, help='Experiment configuration')
    parser.add_argument('--exp', default=None, required=True, help='Experiment name')
    parser.add_argument('--test', action='store_true', default=False, help='Print the number of configurations')
    parser.add_argument('--igeo', type=float, nargs=2, help='Initial lon/lat')
    parser.add_argument('--fgeo', type=float, nargs=2, help='Final lon/lat')
    parser.add_argument('--suff', type=int, default=12, help='Datafile suffix')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')
    args = parser.parse_args()

    coords = np.load(wind_data_path + '/Coords.npy')
    ilon, ilat = args.igeo
    flon, flat = args.fgeo

    if ilat > flat:
        tmp = flat
        flat = ilat
        ilat = tmp

    if ilon > flon:
        tmp = flon
        flon = ilon
        ilon = tmp

    if not(-130 <= ilon <= -63) or not(-130 <= flon <= -63) or not(20 <= ilat <= 50) or not(20 <= flat <= 50):
        raise NameError("Coordinates outside range, use longitude in [-130,-63] and latitude in [20, 50]")

    lsites = [i for i in range(coords.shape[0]) if (ilat <= coords[i][1] <= flat) and (ilon <= coords[i][0] <= flon)]

    config = load_config_file(args.config)

    if args.test:
        print(f"Num Sites= {len(lsites)}")
    else:
        print(f"Num Sites= {len(lsites)}")
        if args.testdb:
            mongoconnection = mongolocaltest

        client = MongoClient(mongoconnection.server)
        db = client[mongoconnection.db]
        if mongoconnection.passwd is not None:
            db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
        col = db[mongoconnection.col + "Test"]

        ids = int(time())

        for site in tqdm(lsites):
            config['site'] = f"{site // 500}-{site}"
            config['data']['datanames'] = [f"{site // 500}-{site}-{args.suff}"]
            config['status'] = 'pending'
            config['experiment'] = args.exp
            config['result'] = []
            config['_id'] = f"{ids}{site:06d}"
            col.insert_one(config)

