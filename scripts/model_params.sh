#!/bin/bash
#SBATCH --job-name=sgd_model_params
#SBATCH --output=/home/aliarab/scratch/sgd/result/model_params/output.log
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --account=def-ester

SLURM_JOB_NAME="sgd_model_params"
source /home/aliarab/miniconda3/bin/activate slice_based
model="model_params"
input_path="/home/aliarab/scratch/sgd/sim_data/data_params/n/500/1"
output_path="/home/aliarab/scratch/sgd/result/model_params/"
project_root="/home/aliarab/src/sgd"

cd $project_root
python sgd.py $model  $input_path $output_path
