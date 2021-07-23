"""
.. module:: Dispatcher

Dispatcher
*************

:Description: Dispatcher

    Jobs dispatcher

:Authors: bejar
    

:Version: 

:Created on: 23/07/2021 7:11 

"""


import argparse
import json
import glob
from Wind.Private.DBConfig import mongolocaltest, mongoconnection
from pymongo import MongoClient
from shutil import copy
from Wind.Config import wind_data_path, bsc_path, jobs_code_path, jobs_root_path, wind_local_jobs_path, wind_jobs_path
from time import strftime, sleep
import os
import sys
from tqdm import tqdm
import numpy as np

__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='convos2s', help='Type of configs')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')
    parser.add_argument('--local', action='store_true', default=False, help='local machine')
    parser.add_argument('--bsc', action='store_true', default=False, help='bsc machines')
    parser.add_argument('--jpw', type=int, default=5, help='jobs per worker')
    parser.add_argument('--workers', type=int, default=5, help='number of bsc workers')
    parser.add_argument('--sleep', type=int, default=30, help='Sleep time between dispatches')

    args = parser.parse_args()

    if args.testdb:
        mongoconnection = mongolocaltest
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    dworkers = {}
    addsleep = 0
    n = 0
    while True:

        if args.local:
            lworkers = glob.glob(wind_local_jobs_path+'/wk*')
        if args.bsc:
            lworkers.extend(glob.glob(wind_jobs_path+'/wk*'))

        for worker in lworkers:
            if worker not in dworkers:
                dworkers[worker] = [0, args.jpw]

        if len(lworkers) != 0:
            query = {'status': 'pending', 'experiment': args.exp}
            lsel = [c for c in col.find(query, limit=args.workers*args.jpw*5)]
            # No more pending configurations
            if len(lsel) == 0:
                for w in dworkers:
                    fconf = open(f"{w}/.end", 'w')
                    fconf.close()
                sys.exit()

            np.random.shuffle(lsel)
            for w in dworkers:
                pending = glob.glob(f'{w}/*.json')
                if len(pending) == 0:
                    dworkers[worker][1] = dworkers[worker][1]+1
                    addsleep = addsleep // 2
                elif (len(pending) > dworkers[worker][1]) and (dworkers[worker][1]>1):
                    dworkers[worker][1] = dworkers[worker][1]-1
                    addsleep += args.sleep
                elif (len(pending) > dworkers[worker][1]) and (dworkers[worker][1]==1):
                    addsleep += args.sleep

                for i in range(dworkers[worker][1]):
                    config = lsel.pop()
                    sconf = json.dumps(config)
                    fconf = open(f"{w}/{config['_id']}.json", 'w')
                    fconf.write(sconf + '\n')
                    fconf.close()
                    col.update_one({'_id': config['_id']}, {'$set': {'status': 'extract'}})

                dworkers[worker][0] += dworkers[worker][1]
                print(f'Worker {w.split("/")[-1]}: A={dworkers[worker][0]} P={len(pending)} step={dworkers[worker][1]}')
        print(f'it {n} - sleep = {args.sleep + addsleep} -----------------------------------------------------------')
        n+=1
        sleep(max(20, args.sleep + addsleep))










