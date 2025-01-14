#!/bin/bash

#SBATCH -A DCCADD
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH -c 8
#SBATCH -N 1
#SBATCH --mem=40G   # Set total memory for the job


export PYTHONUNBUFFERED=FALSE
export PYTHONPATH=/mnt/storage/nobackup/nca121/test_jobs/QM_ESP_Psi4/QM_ESP_Psi4/source:$PYTHONPATH

python /mnt/storage/nobackup/nca121/paper_charge_comparisons/async_chargecraft_more_workers/conformer_test/qc_archive_run/produce_db.py > out.txt