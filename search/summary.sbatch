#!/bin/bash
#SBATCH -p mesonet
#SBATCH --account=m25146
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=s_summary
#SBATCH --output=outputs/search_%x_%j.out
# Stage 5: final tables, figures and the recommended push configuration
source "$ROOT/search/common.sh"
python search/analyse.py summary
