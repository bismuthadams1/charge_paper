#!/bin/bash

#SBATCH -A DCCADD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40gb
#SBATCH --array=30-60

cp -r template_worker batch_worker_$SLURM_ARRAY_TASK_ID
cd batch_worker_$SLURM_ARRAY_TASK_ID

qcfractal-compute-manager --config worker_config.yml