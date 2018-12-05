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
import json
from Wind.Private.DBConfig import mongoconnection
from pymongo import MongoClient
from shutil import copy
from Wind.Config import wind_data_path
from time import strftime
import os

__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nconfig', type=int, default=200, help='number of configs')
    parser.add_argument('--jph', type=int, default=30, help='number of configs')
    parser.add_argument('--exp', default='convos2s', help='number of configs')
    parser.add_argument('--nocopy', action='store_true', default=False, help='copy files')

    args = parser.parse_args()

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    query = {'status': 'pending', 'experiment':args.exp}
    # config = col.find_one(query)

    lconfig = [c for c in col.find(query, limit=args.nconfig)]
    nm = strftime('%Y%m%d%H%M%S')
    os.mkdir(nm)
    os.mkdir('%s/Data' % nm)
    os.mkdir('%s/Jobs' % nm)

    jobtime = (args.nconfig // args.jph) + 2

    batchjob = open('%s/windjob%s.cmd' % (nm, nm), 'w')
    batchjob.write("""#!/bin/bash
# @ job_name = windjob
# @ initialdir = /gpfs/projects/bsc28/bsc28642/Wind/Code/Wind/Experiments
# @ output = windjob%j.out
# @ error = windjob%j.err
# @ total_tasks = 1
# @ gpus_per_node = 1
# @ cpus_per_task = 1
# @ features = k80
""" +
                   "# @ wall_clock_limit = %d:30:00\n" % jobtime
                   +
                   """module purge
                   module purge; module load K80 impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML
                   PYTHONPATH=/gpfs/projects/bsc28/bsc28642/Wind/Code/Wind/
                   export PYTHONPATH
                   
                   """
                   )

    if len(lconfig) > 0:
        for config in lconfig:
            if not args.nocopy:
                print(config['_id'])
                sconf = json.dumps(config)
                fconf = open('./%s/Jobs/' % nm + config['_id'] + '.json', 'w')
                fconf.write(sconf + '\n')
                fconf.close()
                copy(wind_data_path + '/' + config['data']['datanames'][0] + '.npy', './%s/Data/' % nm)
            batchjob.write(
                'python WindExperimentBatch.py --best --early --gpu --mino --config %s\n' % config['_id'])
            col.update({'_id': config['_id']}, {'$set': {'status': 'extract'}})
    batchjob.close()
    print(f"NCONF= {len(lconfig)}")
