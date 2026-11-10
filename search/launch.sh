#!/bin/bash
# Full hyper-parameter search on Juliet. Run from the repository root, on the login node:
#   bash search/launch.sh            (N_CAND=4 by default: number of snapshots refined at stage 4)
# Stages (Slurm dependencies, everything runs unattended):
#   1. train   : 18 runs of search/configs.txt, 300 epochs, G and D saved every 25 epochs  (array)
#   2. snap    : each snapshot -> D fine-tuned 1 epoch, FID at t=0.2 with K=1 and K=10      (array, after its run)
#   3. select  : the N_CAND best (run, epoch) -> results_search/candidates.txt
#   4. gen     : per candidate, search D x K x t x rule, then 3 seeds of the best            (array)
#   5. summary : tables + figures/search_fid_vs_epoch.png + recommended push
# Follow: squeue -u $USER ; results: cat outputs/search_s_summary_*.out
set -e
ROOT=$(pwd)
N_CAND=${N_CAND:-4}
CONFIGS=${CONFIGS:-search/configs.txt}
RES=${RES:-results_search}
[ -f train_search.py ] && [ -f "$CONFIGS" ] || { echo "Run from the repository root."; exit 1; }

# Preflight: everything the compute nodes need (they may have no internet access)
ok=1
for f in eval_classifier.pth incep_real_test.pt venv/bin/activate generate.py evaluate.py finetune_D.py obrs.py dot.py d_models.py model.py utils.py; do
  [ -e "$f" ] || { echo "MISSING: $f"; ok=0; }
done
grep -q "D_arch" finetune_D.py || { echo "finetune_D.py is too old (no --D_arch)"; ok=0; }
grep -q "cnn_sn" d_models.py || { echo "d_models.py is too old (no cnn_sn)"; ok=0; }
grep -q "eigvalsh" evaluate.py || { echo "evaluate.py is too old (CPU FID)"; ok=0; }
grep -q "'01t'" generate.py || { echo "generate.py is too old (no 01t)"; ok=0; }
ls ~/.cache/torch/hub/checkpoints/pt_inception-*.pth >/dev/null 2>&1 \
  || { echo "Inception weights not cached: run once  python -c \"from pytorch_fid.inception import InceptionV3; InceptionV3([3])\""; ok=0; }
[ $ok = 1 ] || { echo "Fix the items above, then relaunch."; exit 1; }

mkdir -p outputs "$RES"
N_RUNS=$(grep -v '^\s*#' "$CONFIGS" | grep -cv '^\s*$')
LAST=$(( N_RUNS - 1 ))
echo "Launching the search: $N_RUNS training runs, $N_CAND candidates refined"

j1=$(sbatch --parsable --export=ALL,ROOT="$ROOT",CONFIGS="$CONFIGS",RES="$RES" --array=0-$LAST search/train.sbatch)
j2=$(sbatch --parsable --export=ALL,ROOT="$ROOT",CONFIGS="$CONFIGS",RES="$RES" --array=0-$LAST --dependency=aftercorr:$j1 search/snap.sbatch)
j3=$(sbatch --parsable --export=ALL,ROOT="$ROOT",CONFIGS="$CONFIGS",RES="$RES",N_CAND="$N_CAND" --dependency=afterany:$j2 search/select.sbatch)
j4=$(sbatch --parsable --export=ALL,ROOT="$ROOT",CONFIGS="$CONFIGS",RES="$RES" --array=0-$(( N_CAND - 1 )) --dependency=afterok:$j3 search/gen.sbatch)
j5=$(sbatch --parsable --export=ALL,ROOT="$ROOT",CONFIGS="$CONFIGS",RES="$RES" --dependency=afterany:$j4 search/summary.sbatch)
echo "jobs: train $j1 | snap $j2 | select $j3 | gen $j4 | summary $j5"
echo "$j1 $j2 $j3 $j4 $j5" > "$RES/job_ids.txt"
echo "Cancel everything: scancel $j1 $j2 $j3 $j4 $j5"
