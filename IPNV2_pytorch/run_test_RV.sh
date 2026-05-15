#!/bin/bash
#SBATCH -n 10
#SBATCH -t 48:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1

#Other SBATCH commands go here
#SBATCH --output=slurm_%j_test.out
#SBATCH --error=slurm_%j_test.err

#Activating conda
#source ~/.bashrc
#conda activate /scratch/yh3529/conda-envs/IPN1_tensorflow

#Your application commands go here
python v2_test_angio.py