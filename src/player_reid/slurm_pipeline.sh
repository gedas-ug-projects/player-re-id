#!/bin/bash
#SBATCH --job-name=player_reid_job    # Job name
#SBATCH --output=output.txt           # Standard output and error log
#SBATCH --error=error.txt             # Error file
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=2             # Number of CPU cores per task
#SBATCH --mem=1M                      # Memory per node
#SBATCH --time=02:00:00               # Time limit hrs:min:sec
#SBATCH --partition=a6000             # Partition name
#SBATCH --gres=gpu:1                  # Request 1 GPU

# Activate Conda environment
source activate reid  # Replace 'myenv' with your environment name

# Run your Python script
python /playpen-storage/levlevi/player-re-id/src/player_reid/pipeline.py

