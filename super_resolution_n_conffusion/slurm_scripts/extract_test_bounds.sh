#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=10g
#SBATCH -c1
#SBATCH --gres=gpu:1,vmem:10g
#SBATCH --array=0-800%100
#SBATCH --requeue
#SBATCH --killable

source ../../venv_conffusion/bin/activate
python3 extract_bounds.py -p test -c config/extract_bounds_16_128_conffusion.json --distributed_worker_id $SLURM_ARRAY_TASK_ID