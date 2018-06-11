"""
.. module:: ExtractConfig

ExtractConfig
*************

:Description: ExtractConfig

    

:Authors: bejar
    

:Version: 

:Created on: 11/06/2018 11:03 

"""

import argparse
from time import time
import json
from Wind.Util import load_config_file
from Wind.Private.DBConfig import mongoconnection
from pymongo import MongoClient

__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False, help='Print the number of configurations')
    parser.add_argument('--iconfig', type=str, help='Initial config')
    parser.add_argument('--fconfig', type=str, help='Final config')
    args = parser.parse_args()


    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    query = {'_id': {'$gte':args.iconfig, '$lte':args.fconfig}}

    qconf = col.find(query)

    if args.test:
        print(len([q for q in qconf]))
    else:
        for config in qconf:
            sconf = json.dumps(config)
            print(config['_id'])
            fconf = open(config['_id']+'.json', 'w')
            fconf.write(sconf + '\n')
            fconf.close()

