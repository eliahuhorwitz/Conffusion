#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=10g
#SBATCH -c1
#SBATCH --gres=gpu:1,vmem:10g
#SBATCH --array=0-1000%100
#SBATCH --requeue
#SBATCH --killable

source ../../venv_conffusion/bin/activate
python3 extract_bounds.py -p calibration -c config/extract_bounds_inpainting_center_conffusion.json --distributed_worker_id $SLURM_ARRAY_TASK_ID