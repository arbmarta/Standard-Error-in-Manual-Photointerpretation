#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:15:00
#SBATCH --partition=debug
#SBATCH --job-name=canopy_metrics
#SBATCH --output=/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Out/canopy_metrics.out
#SBATCH --error=/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Err/canopy_metrics.err

# Run your Python script
module load libspatialindex/1.9.3
source /home/arbmarta/.virtualenvs/myenv/bin/activate
python /scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/canopy_metrics.py
