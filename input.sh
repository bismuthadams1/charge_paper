#!/bin/bash

#SBATCH -A DCCADD
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH -c 32
#SBATCH -N 1
#SBATCH --mem=40gb
#SBATCH -p long
#SBATCH -t 10-00:00:00 

source activate /mnt/nfs/home/nca121/mambaforge/envs/openff_forkedrecharge

export PYTHONUNBUFFERED=FALSE
export PYTHONPATH=/mnt/storage/nobackup/nca121/test_jobs/QM_ESP_Psi4/QM_ESP_Psi4/source:$PYTHONPATH

python ./build_esps.py > out.txt
