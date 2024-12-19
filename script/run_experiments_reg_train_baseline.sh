#!/bin/sh --login
#SBATCH --time=28:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=6  
#SBATCH --mem=64G
#SBATCH --partition=batch 
#SBATCH --job-name=Rate_opti
#SBATCH --output=./bin/%j.out
#SBATCH --error=./bin/%j.out
module purge
eval "$(conda shell.bash hook)"
conda activate torch
python run_experiments_reg_train_baseline.py 
#python main.py