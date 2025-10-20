#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=02:00:00
#SBATCH --job-name=download_meta
#SBATCH --output=/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Download_Meta/download_meta.out
#SBATCH --error=/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Download_Meta/download_meta.err

# Activate virtual environment
source /home/arbmarta/.virtualenvs/myenv/bin/activate

# Run canopy modeling script
python /scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/download_meta.py