"""
.. module:: ChangeThings

RewriteResults
*************

:Description: Remove duplicate experiments

    Changes things in the configuration of all the experiments

    --exp name of the experiment
    --noupdate do not change the database, just count how many experiments will be changed
    --testdb use the test database instead of the final database

:Authors: bejar
    

:Version: 

:Created on: 16/03/2018 13:26 

"""

import argparse
import numpy as np
from Wind.Private.DBConfig import mongoconnection, mongolocaltest
from pymongo import MongoClient


__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='convos2s',  help='Experiment Type')
    parser.add_argument('--status', default=None,  help='Experiment Status')
    parser.add_argument('--noupdate', action='store_true', default=False, help='copy files')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')

    args = parser.parse_args()

    if args.testdb:
        mongoconnection = mongolocaltest
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    if args.status is None:
        configs = col.find({'experiment': args.exp})
    else:
        configs = col.find({'experiment': args.exp, 'status':args.status})


    count = 0
    confs = [conf for conf in configs]
    for conf in confs:
        for conf2 in confs:
            if (conf['_id'] != conf2['_id']) and (conf['site'] == conf2['site']) and (conf['data']['lag'] == conf2['data']['lag']):
                idem = True
                for v in conf['arch']:
                    if conf['arch'][v] != conf2['arch'][v]:
                        idem = False
                        break
                if idem:
                    print(f"Remove Duplicate  {conf['_id']}-{conf2['_id']}")
                    #print(conf['data']['lag'], conf['arch'])
                    #print(conf2['data']['lag'], conf2['arch'])
                    count += 1
                if idem and not args.noupdate:
                    col.remove({'_id': conf2['_id']})

    print(f'{count} Duplicated')
