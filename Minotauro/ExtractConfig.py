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
from Wind.Private.DBConfig import mongolocaltest, mongoconnection
from pymongo import MongoClient
from shutil import copy
from Wind.Config import wind_data_path, bsc_path
from time import strftime
import os

__author__ = 'bejar'


def main(mongoconnection):
    """Extracts configurations from the DB

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--nconfig', type=int, default=200, help='number of configs')
    parser.add_argument('--jph', type=int, default=30, help='Number of jobs per hour')
    parser.add_argument('--exp', default='convos2s', help='Type of configs')
    parser.add_argument('--copy', action='store_true', default=False, help='Copy data files')
    parser.add_argument('--machine', default='mino', help='Machine the scripts are for')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')
    parser.add_argument('--bsc', action='store_true', default=False, help='Copy to bsc path directly')

    args = parser.parse_args()

    if args.testdb:
        mongoconnection = mongolocaltest
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    query = {'status': 'pending', 'experiment': args.exp}
    # config = col.find_one(query)

    lconfig = [c for c in col.find(query, limit=args.nconfig)]
    if not args.bsc:

        nm = strftime('%Y%m%d%H%M%S')
        spath = f"{os.getcwd()}/{nm}"
        os.mkdir(spath)
        os.mkdir(f"{spath}/Data")
        os.mkdir(f"{spath}/Jobs")
        os.mkdir(f"{spath}/Run")
    else:
        nm = strftime('%Y%m%d%H%M%S')
        spath = bsc_path

    jobtime = (args.nconfig // args.jph) + 2

    if args.machine == 'mino':
        jobcontent = f"""#!/bin/bash
# @ job_name = windjob
# @ initialdir = /gpfs/projects/bsc28/bsc28642/Wind/Code/Wind/Experiments
# @ output = /gpfs/projects/bsc28/bsc28642/Wind/Run/windjobmino{nm}.out
# @ error = /gpfs/projects/bsc28/bsc28642/Wind/Run/windjobmino{nm}.err
# @ total_tasks = 1
# @ gpus_per_node = 1
# @ cpus_per_task = 1
# @ features = k80
# @ wall_clock_limit = {jobtime}:30:00
module purge
module load K80 cuda/8.0 mkl/2017.1 CUDNN/5.1.10-cuda_8.0 intel-opencl/2016 python/3.6.0+_ML
PYTHONPATH=/gpfs/projects/bsc28/bsc28642/Wind/Code/Wind/
export PYTHONPATH

"""
    else:
        jobcontent = f"""#!/bin/bash
#SBATCH --job-name="windjob"
#SBATCH -D/gpfs/projects/bsc28/bsc28642/Wind/Code/Wind/Experiments
#SBATCH --output=/gpfs/projects/bsc28/bsc28642/Wind/Run/windjobpower{nm}.out
#SBATCH --error=/gpfs/projects/bsc28/bsc28642/Wind/Run/windjobpower{nm}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time={jobtime}:30:00
#SBATCH --gres=gpu:1
module purge
module load  gcc/6.4.0  cuda/9.1 cudnn/7.1.3 openmpi/3.0.0 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.7 szip/2.1.1 opencv/3.4.1 python/3.6.5_ML
PYTHONPATH=/gpfs/projects/bsc28/bsc28642/Wind/Code/Wind/
export PYTHONPATH

"""

    if args.machine == 'mino':
        batchjob = open(f"{spath}/Run/windjobmino{nm}.cmd", 'w')
        batchjob.write(jobcontent)
    else:
        batchjob = open(f"{spath}/Run/windjobpower{nm}.cmd", 'w')
        batchjob.write(jobcontent)


    if len(lconfig) > 0:
        for config in lconfig:
            print(config['_id'])
            sconf = json.dumps(config)
            fconf = open(f"{spath}/Jobs/{config['_id']}.json", 'w')
            fconf.write(sconf + '\n')
            fconf.close()
            if args.copy:
                copy(f"{wind_data_path}/{config['data']['datanames'][0]}.npy", f"{spath}/Data/")
            if args.machine == 'mino':
                batchjob.write(
                    f"python WindExperimentBatch.py --best --early --gpu --mino --config {config['_id']}\n")
            else:
                batchjob.write(
                    f"mpirun python WindExperimentBatch.py --best --early --gpu --mino --config {config['_id']}\n")

            col.update({'_id': config['_id']}, {'$set': {'status': 'extract'}})
        batchjob.close()

        print(f"NCONF= {len(lconfig)}")


# module load K80 impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

if __name__ == '__main__':
    main(mongoconnection)
