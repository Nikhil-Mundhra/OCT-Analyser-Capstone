#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:2
#SBATCH -c 10

#Other SBATCH commands go here
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

#Activating conda
#source ~/.bashrc

#Your application commands go here
# python v2_train_oct.py
python v2_train.py