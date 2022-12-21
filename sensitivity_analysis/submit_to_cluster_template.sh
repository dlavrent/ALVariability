#!/bin/bash
#SBATCH -n 1
#SBATCH -t 0-01:50
#SBATCH -p serial_requeue
#SBATCH --account=debivort_lab
#SBATCH --mem=64000
#SBATCH --mail-user=dlavrent@g.harvard.edu
#SBATCH --mail-type=ALL

module load Anaconda3/5.0.1-fasrc02
source activate ALVar
python3 run_sim.py