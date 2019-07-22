#!/bin/bash
#SBATCH --job-name="measure"
#SBATCH -D/gpfs/projects/bsc28/bsc28642/Wind/Code/Wind/Scripts/Measutes
#SBATCH --output=/gpfs/projects/bsc28/bsc28642/Wind/Run/measures.out
#SBATCH --error=/gpfs/projects/bsc28/bsc28642/Wind/Run/measures.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=12:00:00
#SBATCH --mem=12000
module purge
module load  gcc/6.4.0  cuda/9.1 cudnn/7.1.3 openmpi/3.0.0 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.7 szip/2.1.1 opencv/3.4.1 python/3.6.5_ML
PYTHONPATH=/gpfs/projects/bsc28/bsc28642/Wind/Code/Wind
export PYTHONPATH

python3 DataMeasutes.py