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
from Wind.Config import wind_data_path, bsc_path, jobs_code_path, jobs_root_path
from time import strftime
import os

__author__ = 'bejar'

# module load K80 cuda/8.0 mkl/2017.1 CUDNN/5.1.10-cuda_8.0 intel-opencl/2016 python/3.6.0+_ML
# module load K80 impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML
# module load  gcc/6.4.0  cuda/9.1 cudnn/7.1.3 openmpi/3.0.0 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.7 szip/2.1.1 ffmpeg/4.0.2 opencv/3.4.1 python/3.6.5_ML


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
    parser.add_argument('--hours', type=int, default=10, help='number of configs')
    parser.add_argument('--mem', type=int, default=2000, help='Memory to use')
    parser.add_argument('--machine', default='mino', help='Machine the scripts are for')
    # parser.add_argument('--bsc', action='store_true', default=False, help='Copy to bsc path directly')

    args = parser.parse_args()

    for nr in range(args.nrep):
        nm = f"{strftime('%Y%m%d%H%M%S')}{nr:03d}"
        if args.machine == 'local':
            spath = '/home/bejar/Wind/'
        else:
            spath =  bsc_path

        os.mkdir(f"{spath}/Jobs/wk{nm}")
            # os.mkdir(f"{spath}/Run")

        jobtime = args.hours

        if jobtime > 47:
            raise NameError(f"{jobtime} is longer than the 48 hours limit")

        if args.machine == 'mino':
            jobcontent = f"""#!/bin/bash
#SBATCH --job-name="{nm}{nr:03d}"
#SBATCH -D{jobs_code_path}/Experiments
#SBATCH --output={jobs_root_path}/Run/windjobmino{nm}{nr:03d}.out
#SBATCH --error={jobs_root_path}/Run/windjobmino{nm}{nr:03d}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time={jobtime}:50:00
#SBATCH --gres=gpu:1
#SBATCH --mem={args.mem}
module purge
module load K80 impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML
PYTHONPATH={jobs_code_path}
export PYTHONPATH

ulimit -s 10240
"""
        elif args.machine == 'local':
            jobcontent = f"""#!/bin/bash
PYTHONPATH=/home/bejar/PycharmProjects/Wind
export PYTHONPATH

cd /home/bejar/PycharmProjects/Wind/Experiments
"""
        else:
            jobcontent = f"""#!/bin/bash
#SBATCH --job-name="{nm}{nr:03d}"
#SBATCH -D{jobs_code_path}/Experiments
#SBATCH --output={jobs_root_path}/Run/windjobpower{nm}{nr:03d}.out
#SBATCH --error={jobs_root_path}/Run/windjobpower{nm}{nr:03d}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time={jobtime}:50:00
#SBATCH --gres=gpu:1
#SBATCH --mem={args.mem}
module purge
module load  gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 python/3.7.4_ML
PYTHONPATH={jobs_code_path}
export PYTHONPATH

ulimit -s 10240
uname -a
"""
        if args.machine == 'mino':
            batchjob = open(f"{spath}/Run/windjobmino{nm}.cmd", 'w')
            batchjob.write(jobcontent)
            batchjob.write("for i in {1..1000}\n")
            batchjob.write("    do\n")
            batchjob.write(
                f"    python WindExperimentWorkerTF1.py --best --early --gpu --mino --jobsdir wk{nm}\n")
        elif args.machine == 'local':
            batchjob = open(f"/{spath}Run/windjoblocal{nm}.cmd", 'w')
            batchjob.write(jobcontent)
            batchjob.write("for i in {1..1000}\n")
            batchjob.write("    do\n")
            batchjob.write(
                f"    python WindExperimentWorker.py --best --early --gpu --local  --jobsdir wk{nm}\n")
        else:
            batchjob = open(f"{spath}/Run/windjobpower{nm}.cmd", 'w')
            batchjob.write(jobcontent)
            batchjob.write("for i in {1..1000}\n")
            batchjob.write("    do\n")
            batchjob.write(
                f"python3 WindExperimentWorker.py --best --early --gpu --mino --gpulog --jobsdir wk{nm}\n")

        batchjob.write(f'    echo \"$i ------------------------------------------------\"\n')
        batchjob.write("done\n")
        batchjob.close()
