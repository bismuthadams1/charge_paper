#!/bin/bash

#SBATCH -A DCCADD
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH -c 32
#SBATCH -N 1
#SBATCH --mem=32gb
#SBATCH -p long

source activate /mnt/nfs/home/nca121/mambaforge/envs/openff_qc

export PYTHONUNBUFFERED=FALSE
export PYTHONPATH=/mnt/storage/nobackup/nca121/test_jobs/QM_ESP_Psi4/QM_ESP_Psi4/source:$PYTHONPATH

python ./build_esps.py > out_2.txt
