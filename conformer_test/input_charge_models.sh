#!/bin/bash

#SBATCH -A DCCADD
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH -c 40
#SBATCH -N 1
#SBATCH -p bigmem
#SBATCH --mem=500gb

source activate /mnt/nfs/home/nca121/mambaforge/envs/charge_model_env

export PYTHONUNBUFFERED=FALSE
export PYTHONPATH=/mnt/storage/nobackup/nca121/test_jobs/QM_ESP_Psi4/QM_ESP_Psi4/source:$PYTHONPATH

python ./build_charge_models.py > build_charge_models.txt
