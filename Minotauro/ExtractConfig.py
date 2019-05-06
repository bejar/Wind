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
from Wind.Config import wind_data_path, bsc_path, jobs_code_path, jobs_root_path
from time import strftime
import os
from tqdm import tqdm
import numpy as np

__author__ = 'bejar'

# module load K80 cuda/8.0 mkl/2017.1 CUDNN/5.1.10-cuda_8.0 intel-opencl/2016 python/3.6.0+_ML
# module load K80 impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML


# jobcontentmino= f"""#!/bin/bash
# # @ job_name = windjob
# # @ initialdir = {jobs_code_path}/Experiments
# # @ output = {jobs_root_path}/Run/windjobmino{nm}{nr:03d}.out
# # @ error = {jobs_root_path}/Run/windjobmino{nm}{nr:03d}.err
# # @ total_tasks = 1
# # @ gpus_per_node = 1
# # @ cpus_per_task = 1
# # @ features = k80
# # @ wall_clock_limit = {jobtime}:50:00
# module purge
# module load K80 impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML
# PYTHONPATH={jobs_code_path}
# export PYTHONPATH
#
# """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrep', type=int, default=1, help='number of replicas of the script')
    parser.add_argument('--nconfig', type=int, default=200, help='number of configs')
    parser.add_argument('--jph', type=int, default=30, help='Number of jobs per hour')
    parser.add_argument('--mem', type=int, default=2000, help='Memory to use')
    parser.add_argument('--exp', default='convos2s', help='Type of configs')
    parser.add_argument('--copy', action='store_true', default=False, help='Copy data files')
    parser.add_argument('--machine', default='mino', help='Machine the scripts are for')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')
    parser.add_argument('--bsc', action='store_true', default=False, help='Copy to bsc path directly')
    parser.add_argument('--rand', action='store_true', default=False, help='Add some randomness to the selection of sites')

    args = parser.parse_args()

    if args.testdb:
        mongoconnection = mongolocaltest
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    for nr in range(args.nrep):
        query = {'status': 'pending', 'experiment': args.exp}
        # config = col.find_one(query)

        if args.rand:
            lsel = [c for c in col.find(query, limit=args.nconfig*100)]
            np.random.shuffle(lsel)
            if len(lsel)> args.nconfig:
               lconfig = [c for c in np.random.choice(lsel, args.nconfig, replace=False)]
            else:
               lconfig = lsel
        else:
            lconfig = [c for c in col.find(query, limit=args.nconfig)]

        if len(lconfig) == 0:
            raise NameError('No configurations found')

        # Creates a directory with Data/Jobs/Scripts in the current path
        if not args.bsc:
            nm = strftime('%Y%m%d%H%M%S')
            spath = f"{os.getcwd()}/{nm}"
            os.mkdir(spath)
            os.mkdir(f"{spath}/Data")
            os.mkdir(f"{spath}/Jobs")
            os.mkdir(f"{spath}/Run")
        # Copies everything to the remotely mounted BSC dir
        else:
            nm = strftime('%Y%m%d%H%M%S')
            spath = bsc_path

        jobtime = (len(lconfig) // args.jph) + 1

        if jobtime > 47:
            raise NameError(f"{jobtime} is longer than the 48 hours limit, reduce number of configs")

        if args.machine == 'mino':
            jobcontent = f"""#!/bin/bash
#SBATCH --job-name="windjob"
#SBATCH -D{jobs_code_path}/Experiments
#SBATCH --output={jobs_root_path}/Run/windjobmino{nm}{nr:03d}.out
#SBATCH --error={jobs_root_path}/Run/windjobmino{nm}{nr:03d}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time={jobtime}:50:00
#SBATCH --gres=gpu:1
#SBATCH --mem={args.mem}
module purge
module load K80 impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML
PYTHONPATH={jobs_code_path}
export PYTHONPATH

"""

        else:
            jobcontent = f"""#!/bin/bash
#SBATCH --job-name="windjob"
#SBATCH -D{jobs_code_path}/Experiments
#SBATCH --output={jobs_root_path}/Run/windjobpower{nm}{nr:03d}.out
#SBATCH --error={jobs_root_path}/Run/windjobpower{nm}{nr:03d}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time={jobtime}:50:00
#SBATCH --gres=gpu:1
#SBATCH --mem={args.mem}
module purge
module load  gcc/6.4.0  cuda/9.1 cudnn/7.1.3 openmpi/3.0.0 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.7 szip/2.1.1 opencv/3.4.1 python/3.6.5_ML
PYTHONPATH={jobs_code_path}
export PYTHONPATH

"""

        if args.machine == 'mino':
            batchjob = open(f"{spath}/Run/windjobmino{nm}{nr:03d}.cmd", 'w')
            batchjob.write(jobcontent)
        else:
            batchjob = open(f"{spath}/Run/windjobpower{nm}{nr:03d}.cmd", 'w')
            batchjob.write(jobcontent)

        if len(lconfig) > 0:
            for config in tqdm(lconfig):
                # print(config['_id'])
                config['host'] = args.machine
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
                        f"python3 WindExperimentBatch.py --best --early --gpu --mino --config {config['_id']}\n")
                        #f"mpirun python WindExperimentBatch.py --best --early --gpu --gpulog --mino --config {config['_id']}\n")

                col.update({'_id': config['_id']}, {'$set': {'status': 'extract'}})
            batchjob.close()
            print(f"\nScript {nr}: Estimated running time = {jobtime} hours")
            # print(f"NCONF= {len(lconfig)}")
