"""
.. module:: RewriteResults

RewriteResults
*************

:Description: RewriteResults

    Changes the status of an experiment with results lower than a threshold to pending so it can be run again

    --thres threshold to use (lower or equal than that)
    --exp name of the experiment
    --status status of the experiment
    --noupdate do not change the database, just count how many experiments are below the threshold
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
    parser.add_argument('--thres', type=float, default=2.0,  help='Rewriting threshold')    
    parser.add_argument('--exp', default='convos2s',  help='Experiment Type')    
    parser.add_argument('--status', default='done',  help='Experiment status')
    parser.add_argument('--noupdate', action='store_true', default=False, help='Do not change anything')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')

    args = parser.parse_args()

    if args.testdb:
        mongoconnection = mongolocaltest
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    configs = col.find({'experiment': args.exp, 'status': args.status})

    count = 0
    for conf in configs:
        # print(conf['site'])
        data = np.array(conf['result'])
        vsum = np.sum(data[:, 1])
        if vsum <= args.thres:
            print(conf['site'], vsum)
            if not args.noupdate:
                col.update_one({'_id': conf['_id']}, {'$set': {'status': 'pending'}})
            count += 1

    print(count)
