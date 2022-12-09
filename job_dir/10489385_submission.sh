#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=3
#SBATCH --error=/home/mvetter/learnable_grid_embeddings/job_dir/%j_0_log.err
#SBATCH --gpus-per-node=2
#SBATCH --job-name=mae
#SBATCH --mem=80GB
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --open-mode=append
#SBATCH --output=/home/mvetter/learnable_grid_embeddings/job_dir/%j_0_log.out
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --signal=USR2@120
#SBATCH --time=600
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/mvetter/learnable_grid_embeddings/job_dir/%j_%t_log.out --error /home/mvetter/learnable_grid_embeddings/job_dir/%j_%t_log.err /sw/arch/Debian10/EB_production/2022/software/Python/3.10.4-GCCcore-11.3.0-bare/bin/python3 -u -m submitit.core._submit /home/mvetter/learnable_grid_embeddings/job_dir
