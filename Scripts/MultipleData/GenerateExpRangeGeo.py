"""
.. module:: GenerateExpRangeSections

GenerateExpSites
*************

:Description: GenerateRangeSections

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
from Wind.Private.DBConfig import mongoconnection
from pymongo import MongoClient
from Wind.Config.Paths import wind_data_path
import numpy as np
__author__ = 'bejar'


def main():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configbatchregdir', help='Experiment configuration')
    parser.add_argument('--test', action='store_true', default=False, help='Print the number of configurations')
    parser.add_argument('--igeo', type=int, help='Initial lat/lon')
    parser.add_argument('--fgeo', type=int, help='Final lat/lon')
    parser.add_argument('--suff', type=int, default=12, help='Datafile suffix')
    args = parser.parse_args()
    coords = np.load(wind_data_path + '/coords.npy')

    config = load_config_file(args.config)

    if args.test:
        print(500 * (args.fsec - args.isec + 1))
    else:
        print(500 * (args.fsec - args.isec + 1))
        client = MongoClient(mongoconnection.server)
        db = client[mongoconnection.db]
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
        col = db[mongoconnection.col]

        ids = int(time())
        for i, sec in enumerate(range(args.isec, args.fsec + 1)):
            for site in range(i * 500, (i + 1) * 500):
                config['site'] = f"{sec}-{site}"
                config['data']['datanames'] = [f"{sec}-{site}-{args.suff}"]
                config['status'] = 'pending'
                config['result'] = []
                config['_id'] = f"{ids}{site:06d}"
                col.insert_one(config)

if __name__ == '__main__':
    main()
