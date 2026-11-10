#!/bin/bash
#SBATCH -p mesonet
#SBATCH --account=m25146
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=s_train
#SBATCH --output=outputs/search_%x_%A_%a.out
# Stage 1: one training run per array task (line SLURM_ARRAY_TASK_ID of configs.txt).
source "$ROOT/search/common.sh"
name=$(run_name "$SLURM_ARRAY_TASK_ID"); opts=$(run_options "$SLURM_ARRAY_TASK_ID")
echo "### train $name : $opts"
if [ -f "checkpoints/search/$name/G_ep300.pth" ]; then echo "already trained, skipping"; exit 0; fi
python train_search.py $opts --epochs 300 --snap_every 25 --out "search/$name" || echo "training failed (snapshots saved so far are still evaluated)"
exit 0   # keep the aftercorr dependency of the snap stage satisfiable
