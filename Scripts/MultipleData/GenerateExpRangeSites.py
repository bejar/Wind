"""
.. module:: GenerateExpSites

GenerateExpSites
*************

:Description: GenerateExpSites

    

:Authors: bejar
    

:Version: 

:Created on: 07/06/2018 15:45 

"""

import argparse
from time import time

from Wind.Util import load_config_file
from Wind.Private.DBConfig import mongoconnection
from pymongo import MongoClient

__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configbatchregdir', help='Experiment configuration')
    parser.add_argument('--test', action='store_true', default=False, help='Print the number of configurations')
    parser.add_argument('--sec', type=int, help='Section')
    parser.add_argument('--isite', type=int, help='Initial Site')
    parser.add_argument('--fsite', type=int, help='Final Site')
    parser.add_argument('--suff', type=int, default=12, help='Datafile suffix')
    args = parser.parse_args()

    config = load_config_file(args.config)

    if args.test:
        print(args.fsite - args.isite + 1)
    else:
        print(args.fsite - args.isite + 1)
        client = MongoClient(mongoconnection.server)
        db = client[mongoconnection.db]
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
        col = db[mongoconnection.col]

        ids = int(time())
        for j, site in enumerate(range(args.isite, args.fsite + 1)):
            config['site'] = '%d-%d' % (args.sec, site)
            config['data']['datanames'] = ['%d-%d-%d' % (args.sec, site, args.suff)]
            config['status'] = 'pending'
            config['result'] = []
            config['_id'] = "%d%06d" % (ids, site)
            col.insert_one(config)
            # print('%d-%d-%d' % (sec, site, args.suff))
            # print(config)
