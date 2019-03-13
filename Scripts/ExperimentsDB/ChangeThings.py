"""
.. module:: ChangeThings

RewriteResults
*************

:Description: RewriteResults

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
    for conf in configs:
        if not args.noupdate:
            col.remove({'_id': conf['_id']})
                #col.update({'_id': conf['_id']},{'$set': {'experiment':'mlps2selu' }})
#                col.update({'_id': conf['_id']}, {'$set': {'arch.filters': [128,128,128]}})
#                col.update({'_id': conf['_id']}, {'$set': {'status': 'pending'}})
            count += 1

    print(f'{count} Experiments changed')
