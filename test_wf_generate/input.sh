#!/bin/bash

#SBATCH -A DCCADD
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH -c 8
#SBATCH -N 1
#SBATCH --mem=32gb

source activate /mnt/nfs/home/nca121/mambaforge/envs/openff_forkedrecharge

export PYTHONUNBUFFERED=FALSE
export PYTHONPATH=/mnt/storage/nobackup/nca121/test_jobs/QM_ESP_Psi4/QM_ESP_Psi4/source:$PYTHONPATH

python /mnt/storage/nobackup/nca121/QC_archive_50K_esp_gen/async_chargecraft/test_wf_generate/isolated_wf.py > out.txt
