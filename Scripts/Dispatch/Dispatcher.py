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
from Wind.Config import wind_data_path, jobs_root_path, wind_local_jobs_path, wind_jobs_path, wind_local_res_path, \
    wind_res_path
from time import strftime, sleep, ctime
import os
import sys
from Wind.Misc import load_config_file
import numpy as np

__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='convos2s', help='Type of configs')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')
    parser.add_argument('--local', action='store_true', default=False, help='local machine')
    parser.add_argument('--bsc', action='store_true', default=False, help='bsc machines')
    parser.add_argument('--noup', action='store_true', default=False, help='bsc machines')
    parser.add_argument('--jpw', type=int, default=5, help='jobs per worker')
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
    n = 0
    served = 0
    finish = False

    if args.local:
        lworkers = glob.glob(wind_local_jobs_path + '/wk*')

    if args.bsc:
        lworkers.extend(glob.glob('/home/bejar/bsc/Wind/Jobs/wk*'))

    # for w in lworkers:
    #    if os.path.exists(w+'/.end'):
    #        os.remove(w+'/.end')  

    while True:
        wdone = False
        if args.local:
            lworkers = glob.glob(wind_local_jobs_path + '/wk*')

        if args.bsc:
            lworkers.extend(glob.glob('/home/bejar/bsc/Wind/Jobs/wk*'))

        for worker in lworkers:
            if worker not in dworkers:
                dworkers[worker] = [0]

        if len(lworkers) != 0:
            query = {'status': 'pending', 'experiment': args.exp}
            lsel = [c for c in col.find(query, limit=args.jpw * 100)]
            # No more pending configurations
            if len(lsel) == 0:
                for w in dworkers:
                    fconf = open(f"{w}/.end", 'w')
                    fconf.close()
                    finish = True
                    wdone = True

            if not finish:
                np.random.shuffle(lsel)
                for w in dworkers:
                    pending = glob.glob(f'{w}/*.json')

                    diff = args.jpw - len(pending)
                    if diff > 0:
                        wdone = True
                        for i in range(diff):
                            if len(lsel) > 0:
                                config = lsel.pop()
                            else:
                                break
                            sconf = json.dumps(config)
                            fconf = open(f"{w}/{config['_id']}.json", 'w')
                            fconf.write(sconf + '\n')
                            fconf.close()
                            col.update_one({'_id': config['_id']}, {'$set': {'status': 'extract'}})
                            dworkers[w][0] += 1
                            served += 1
                    print(f'Worker {w.split("/")[-1]}: A={dworkers[w][0]}')

                    done = glob.glob(f'{w}/*.done')
                    for d in done:
                        os.remove(f'{d}')

        # Upload Results
        lres = []
        still = False
        if wdone and not args.noup:
            if args.local:
                lres.extend(glob.glob(wind_local_res_path + '/res*.json'))
            if args.bsc:
                lres.extend(glob.glob('/home/bejar/bsc/Wind/Res/res*.json'))
            for file in lres:
                still = True
                config = load_config_file(file, upload=True, abspath=True)
                exists = col.find_one({'_id': config['_id']})
                if exists:
                    col.update_one({'_id': config['_id']}, {'$set': {'status': 'done'}})
                    if 'results' in config:
                        col.update_one({'_id': config['_id']}, {'$set': {'result': config['results']}})
                    elif 'result' in config:
                        col.update_one({'_id': config['_id']}, {'$set': {'result': config['result']}})
                    col.update_one({'_id': config['_id']}, {'$set': {'etime': config['etime']}})
                    if 'btime' in config:
                        col.update_one({'_id': config['_id']}, {'$set': {'btime': config['btime']}})
                    else:
                        col.update_one({'_id': config['_id']}, {'$set': {'btime': config['etime']}})
                os.rename(file, f'{file.replace(".json", ".done")}')

            for file in lres:
                os.remove(f'{file.replace(".json", ".done")}')

        print(f'it {n} - served= {served} uploaded = {len(lres)} - {ctime()} --------------------------', flush=True)
        n += 1

        if finish and not still:
            sys.exit()

        sleep(args.sleep)
