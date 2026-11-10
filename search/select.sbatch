#!/bin/bash
#SBATCH -p mesonet
#SBATCH --account=m25146
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=s_select
#SBATCH --output=outputs/search_%x_%j.out
# Stage 3: rank all snapshots, write the N best (run, epoch) to results_search/candidates.txt
source "$ROOT/search/common.sh"
python search/analyse.py select --n "${N_CAND:-4}"
