#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=02:00:00
#SBATCH --job-name=canopy_metrics
#SBATCH --output=/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Out/canopy_metrics.out
#SBATCH --error=/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Err/canopy_metrics.err

# Run your Python script
source /home/arbmarta/.virtualenvs/myenv/bin/activate
python /scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/canopy_metrics.py
