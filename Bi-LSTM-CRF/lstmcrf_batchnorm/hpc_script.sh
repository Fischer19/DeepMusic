#!/bin/sh
#
#SBATCH --verbose
#SBATCH -p gpu 
#SBATCH --job-name=batch_bilstmcrf_toy_data
#SBATCH --output=batch_bilstmcrf_toy_data%j.out
#SBATCH --error=batch_bilstmcrf_toy_data%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yg1227@nyu.edu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --exclusive

echo "SLURM_JOBID="#SLURM_JOBID
module load anaconda3
module load cuda/9.0
source activate rl
python lstmcrf_skip.py
echo "All Done!"
