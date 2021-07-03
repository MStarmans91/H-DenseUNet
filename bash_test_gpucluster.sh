#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=14G
#SBATCH --gres=gpu:GTX:1
#SBATCH -p GPU
#SBATCH -t 00:10:00
#SBATCH -o /home/mstarmans/Logs/out_%j.log
#SBATCH -e /home/mstarmans/Logs/error_%j.log

# Load the modules
module purge
module load python/2.7.15
module load tensorflow/1.10.0

source $HOME/VEHDensU/bin/activate

python $HOME/git/H-DenseUNet/segment_patient.py
