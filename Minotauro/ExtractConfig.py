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

__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nconfig', type=int, help='number of configs')
    args = parser.parse_args()


    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]


    query = {'status': 'pending'}
    # config = col.find_one(query)

    lconfig = [c for c in col.find(query, limit=args.nconfig)]

    batchjob = open('windjob%s.cmd'%strftime('%Y%m%d%H%M'),'w')
    batchjob.write("""#!/bin/bash
# @ job_name = windjob
# @ initialdir = /gpfs/projects/nct00/nct00001/DLMAI/Wind/Experiments
# @ output = windjob%j.out
# @ error = windjob%j.err
# @ total_tasks = 1
# @ gpus_per_node = 1
# @ cpus_per_task = 1
# @ features = k80
# @ wall_clock_limit = 15:00:00

module purge
module load K80 cuda/8.0 mkl/2017.1 CUDNN/5.1.10-cuda_8.0 intel-opencl/2016 python/3.6.0+_ML
PYTHONPATH=/gpfs/projects/nct00/nct00001/DLMAI/Wind
export PYTHONPATH

""")

    if len(lconfig) > 0:
        for config in lconfig:
            sconf = json.dumps(config)
            print(config['_id'])
            fconf = open(config['_id']+'.json', 'w')
            fconf.write(sconf + '\n')
            fconf.close()
            copy(wind_data_path +'/'+ config['data']['datanames'][0]+'.npy', './Data/')
            batchjob.write(
                'python WindPredictionBatch.py --best --early --gpu --mino --config %s >res.out\n' % config['_id'])
            col.update({'_id': config['_id']}, {'$set': {'status': 'extract'}})
    batchjob.close()

