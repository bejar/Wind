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
from time import time
import numpy as np
from Wind.Util import load_config_file
from Wind.Data import generate_dataset
from Wind.Private.DBConfig import mongoconnection
from copy import deepcopy
from pymongo import MongoClient
import json

__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thres', type=float, default=2.0,  help='Rewriting threshold')    
    parser.add_argument('--exp', default='convos2s',  help='Experiment Type')    
    parser.add_argument('--status', default='done',  help='Experiment status')    
    args = parser.parse_args()

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    #    configs = col.find({'experiment':'mlpregs2s','status':'working' })
    configs = col.find({'experiment': args.exp, 'status': args.status})

    count = 0
    for conf in configs:
        # print(conf['site'])
        data = np.array(conf['result'])
        vsum = np.sum(data[:, 1])
        if vsum < args.thres:
            print(conf['site'], vsum)
            col.update({'_id': conf['_id']}, {'$set': {'status': 'pending'}})
            count += 1

    print(count)
