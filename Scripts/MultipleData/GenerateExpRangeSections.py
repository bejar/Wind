"""
.. module:: GenerateExpRangeSections

GenerateExpSites
*************

:Description: GenerateRangeSections

    Generates and uploads to the DB configurations using --config configuration
    it begin at files section --isec and ends at section --fsec
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

# sections 0-253

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, required=True, help='Experiment configuration')
    parser.add_argument('--exp', default=None, required=True, help='Experiment name')
    parser.add_argument('--test', action='store_true', default=False, help='Print the number of configurations')
    parser.add_argument('--isec', type=int, required=True, help='Initial section')
    parser.add_argument('--fsec', type=int, required=True, help='Final section')
    parser.add_argument('--suff', type=int, default=12, help='Datafile suffix')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')

    args = parser.parse_args()

    if args.fsec > 253 or args.isec > 253 :
        raise NameError("The numer of sections can not be larger than 253")

    config = load_config_file(args.config)
    if args.test:
        print(500 * (args.fsec - args.isec + 1))
    else:
        print(500 * (args.fsec - args.isec + 1))
        if args.testdb:
            mongoconnection = mongolocaltest
        client = MongoClient(mongoconnection.server)
        db = client[mongoconnection.db]
        if mongoconnection.passwd is not None:
            db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
        col = db[mongoconnection.col]

        ids = int(time())
        for i, sec in tqdm(enumerate(range(args.isec, args.fsec + 1))):
            for site in range(i * 500, (i + 1) * 500):
                config['data']['datanames'] = [f"{sec}-{site}-{args.suff}"]
                site = config['data']['datanames'][0].split('-')
                config['site'] = '-'.join(site[:2])
                config['status'] = 'pending'
                config['experiment'] = args.exp
                config['result'] = []
                config['_id'] = f"{ids}{i:05d}{int(site[1]):06d}"
                col.insert_one(config)
