"""
.. module:: UploadResults

UploadResults
*************

:Description: UploadResults

    

:Authors: bejar
    

:Version: 

:Created on: 16/03/2018 13:26 

"""

import argparse
import numpy as np
from Wind.Private.DBConfig import mongoconnection
from pymongo import MongoClient


__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thres', type=float, default=2.0,  help='Rewriting threshold')    
    parser.add_argument('--exp', default='convos2s',  help='Experiment Type')    
    parser.add_argument('--status', default='done',  help='Experiment status')
    parser.add_argument('--noupdate', action='store_true', default=False, help='copy files')
    args = parser.parse_args()

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    configs = col.find({'experiment': args.exp, 'status': args.status})

    count = 0
    for conf in configs:
        # print(conf['site'])
        data = np.array(conf['result'])
        vsum = np.sum(data[:, 1])
        if vsum < args.thres:
            print(conf['site'], vsum)
            if not args.noupdate:
                col.update({'_id': conf['_id']}, {'$set': {'status': 'pending'}})
            count += 1

    print(count)
