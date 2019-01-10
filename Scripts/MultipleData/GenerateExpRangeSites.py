"""
.. module:: GenerateExpRangeSites

GenerateExpSites
*************

:Description: GenerateRangeExpSites

    Generates and uploads to the DB configurations using --config configuration
    it begins at file --isite and ends at section --fsite
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
from tqdm import tqdm

__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configbatchregdir', help='Experiment configuration')
    parser.add_argument('--test', action='store_true', default=False, help='Print the number of configurations')
    parser.add_argument('--isite', type=int, help='Initial Site')
    parser.add_argument('--fsite', type=int, help='Final Site')
    parser.add_argument('--suff', type=int, default=12, help='Datafile suffix')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')
    args = parser.parse_args()

    config = load_config_file(args.config)

    if args.test:
        print(args.fsite - args.isite + 1)
    else:
        if args.testdb:
            mongoconnection = mongolocaltest
        client = MongoClient(mongoconnection.server)
        db = client[mongoconnection.db]
        if mongoconnection.passwd is not None:
            db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
        col = db[mongoconnection.col]

        ids = int(time())
        for site in tqdm(range(args.isite, args.fsite + 1)):
            config['site'] = f'{site//500}-{site}'
            config['data']['datanames'] = [f'{site//500}-{site}-{args.suff}']
            config['status'] = 'pending'
            config['result'] = []
            config['_id'] = f"{ids}{site}"
            col.insert_one(config)

