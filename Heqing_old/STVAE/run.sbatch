#!/bin/bash

# This script will request one GPU device and 1 CPU core

#SBATCH --job-name=job1111
#SBATCH --output=aff.out
#SBATCH --error=aff.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu2				
#SBATCH --gres=gpu:1

module load cuda/9.0
module load Anaconda3/5.0.1
source activate pytorch_cuda_9.0
module unload gcc

python3 main.py --sdim=30 --nepoch=40 --gpu
