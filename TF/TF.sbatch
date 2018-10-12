#!/bin/bash

# This script will request one GPU device
#SBATCH --account=pi-yaliamit
#SBATCH --job-name=TF_test
#SBATCH --output=TF_test.out
#SBATCH --error=TF_test.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu2             
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --exclusive
module load cuda/9.0
module load Anaconda3/5.0.1
source activate DL_GPU_cuda_9.0
CUDA_LAUNCH_BLOCKING=2 device=gpu _scripts/runthings
