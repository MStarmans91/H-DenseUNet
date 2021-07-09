#!/bin/bash
#SBATCH -N 1      # request 5 nodes
#SBATCH -p gpu    # request partition 'short', see below
#SBATCH -n 1      # request 40 processes (each runs on 1 core), mostly to run MPI programs on  SLURM will compute the number of nodes needed
#SBATCH -t 2-00:00:00  # The job can take at most 48 wall-clock hours.
module load 2019
module load Python/2.7.15-intel-2018b
source $HOME/VEHDensU/bin/activate
module load pre2019
module load cuda/8.0.44
module load cudnn/8.0-v5.1
python $HOME/git/HDenseUNet/segment_patient.py
