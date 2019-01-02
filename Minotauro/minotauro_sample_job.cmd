#!/bin/bash
# @ job_name = windjob
# @ initialdir = /gpfs/projects/bsc28/bsc28642/Wind/Code/Wind/Experiments
# @ output = windjob%j.out
# @ error = windjob%j.err
# @ total_tasks = 1
# @ gpus_per_node = 1
# @ cpus_per_task = 1
# @ features = k80
# @ wall_clock_limit = 10:30:00

module purge
module load K80 impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML
PYTHONPATH=/gpfs/projects/bsc28/bsc28642/Wind/Code/Wind
export PYTHONPATH
python WindExperimentBatch.py --best --early --gpu --mino --config
