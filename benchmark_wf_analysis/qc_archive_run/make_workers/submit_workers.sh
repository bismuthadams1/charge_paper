#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40gb
#SBATCH --array=30-60
#SBATCH --partition=cpu

cp -r template_worker batch_worker_$SLURM_ARRAY_TASK_ID
cd batch_worker_$SLURM_ARRAY_TASK_ID

mamba run -n qcarchive-worker-openff-psi4 qcfractal-compute-manager --config worker_config.yml