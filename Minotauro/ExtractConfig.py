"""
.. module:: ExtractConfig

ExtractConfig
*************

:Description: ExtractConfig

    Extracts a list of configurations from mongoDB, copies the data (if told so) and generates a script to be
    executed at minotauro

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


def main():
    """Extracts configurations from the DB

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--nconfig', type=int, default=200, help='number of configs')
    parser.add_argument('--jph', type=int, default=30, help='Number of jobs per hour')
    parser.add_argument('--exp', default='convos2s', help='Type of configs')
    parser.add_argument('--nocopy', action='store_true', default=False, help='copy files')
    parser.add_argument('--machine', default='mino', help='Machine the scripts are for')


    args = parser.parse_args()

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    query = {'status': 'pending', 'experiment': args.exp}
    # config = col.find_one(query)

    lconfig = [c for c in col.find(query, limit=args.nconfig)]
    nm = strftime('%Y%m%d%H%M%S')
    os.mkdir(nm)
    os.mkdir(f"{nm}/Data")
    os.mkdir(f"{nm}/Jobs")

    jobtime = (args.nconfig // args.jph) + 2

    if args.machine == 'mino':

        batchjob = open(f"'{nm}/windjob{nm}.cmd'", 'w')
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
                       f"# @ wall_clock_limit = {jobtime}:30:00\n"
                       +
                       """module purge
                       module load K80 cuda/8.0 mkl/2017.1 CUDNN/5.1.10-cuda_8.0 intel-opencl/2016 python/3.6.0+_ML
                       PYTHONPATH=/gpfs/projects/bsc28/bsc28642/Wind/Code/Wind/
                       export PYTHONPATH
                       
                       """)

        if len(lconfig) > 0:
            for config in lconfig:
                if not args.nocopy:
                    print(config['_id'])
                    sconf = json.dumps(config)
                    fconf = open(f"./{nm}/Jobs/{config['_id']}.json", 'w')
                    fconf.write(sconf + '\n')
                    fconf.close()
                    copy(f"{wind_data_path}/{config['data']['datanames'][0]}.npy", f"./{nm}/Data/")
                batchjob.write(
                    f"python WindExperimentBatch.py --best --early --gpu --mino --config {config['_id']}\n")
                col.update({'_id': config['_id']}, {'$set': {'status': 'extract'}})
        batchjob.close()


        print(f"NCONF= {len(lconfig)}")


# module load K80 impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

if __name__ == '__main__':
    main()
