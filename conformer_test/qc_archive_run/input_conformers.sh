#!/bin/bash

#SBATCH -A DCCADD
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH -c 44
#SBATCH -N 1
##SBATCH -p bigmem
##SBATCH --mem=500gb

export PYTHONUNBUFFERED=FALSE
export PYTHONPATH=/mnt/storage/nobackup/nca121/test_jobs/QM_ESP_Psi4/QM_ESP_Psi4/source:$PYTHONPATH
export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3= 1

source activate /mnt/nfs/home/nca121/.bashrc

mamba run -n charge_model_env python /mnt/storage/nobackup/nca121/paper_charge_comparisons/async_chargecraft_more_workers/conformer_test/qc_archive_run/build_charge_models.py > out.txt