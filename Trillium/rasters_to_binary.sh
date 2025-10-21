#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:40:00
#SBATCH --job-name=rasters_to_binary
#SBATCH --output=/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Out/rasters_to_binary.out
#SBATCH --error=/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Err/rasters_to_binary.err

# Run your Python script
source /home/arbmarta/.virtualenvs/myenv/bin/activate
python /scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/rasters_to_binary.py
