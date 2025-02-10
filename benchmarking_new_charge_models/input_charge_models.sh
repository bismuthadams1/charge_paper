#!/bin/bash

#SBATCH -A DCCADD
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH -c 32
#SBATCH -N 1
#SBATCH --mem=40gb

source activate /mnt/nfs/home/nca121/mambaforge/envs/charge_model_env
export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3= 1
export PYTHONUNBUFFERED=FALSE
export PYTHONPATH=/mnt/storage/nobackup/nca121/test_jobs/QM_ESP_Psi4/QM_ESP_Psi4/source:$PYTHONPATH

mamba run -n charge_model_env python ./build_charge_models.py > build_charge_models.txt
