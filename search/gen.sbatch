#!/bin/bash
#SBATCH -p mesonet
#SBATCH --account=m25146
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --job-name=s_gen
#SBATCH --output=outputs/search_%x_%A_%a.out
# Stage 4: generation-side coordinate search for one candidate (run, epoch), at seed 0:
#   D in {co-trained, fine-tuned 1 epoch, fine-tuned 3 epochs}  (K=10, t=0.2)
#   K in {1, 2, 5, 10, 20, 50}                                  (best D, t=0.2)
#   t in {0.12, 0.16, 0.2, 0.25, 0.3}                           (best D, best K)
#   rule: OBRS vs deterministic top-1/K                         (best D, K, t)
# then the best configuration on seeds 0, 1, 2 (step=final).
source "$ROOT/search/common.sh"
line=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" "$ROOT/$RES/candidates.txt")
[ -n "$line" ] || { echo "no candidate $SLURM_ARRAY_TASK_ID"; exit 0; }
read -r name ep <<< "$line"
csv="$ROOT/$RES/gen_${name}_ep$ep.csv"
arch=$(run_arch "$name")
workdir "gen_${name}_ep$ep"
G="search/$name/G_ep$ep.pth"
declare -A DPATH=( [co]="search/$name/D_ep$ep.pth" [ft1]="search/$name/Dft1_ep$ep.pth" [ft3]="search/$name/Dft3_ep$ep.pth" )
[ -f "checkpoints/${DPATH[ft1]}" ] || python finetune_D.py --D_arch "$arch" --epochs 1 --G_name "$G" --D_init "${DPATH[co]}" --out_name "${DPATH[ft1]}"
[ -f "checkpoints/${DPATH[ft3]}" ] || python finetune_D.py --D_arch "$arch" --epochs 3 --G_name "$G" --D_init "${DPATH[co]}" --out_name "${DPATH[ft3]}"
base="run=$name;ep=$ep"
pick () { python "$ROOT/search/analyse.py" pick --csv "$csv" --step "$1" --key "$2"; }

for d in co ft1 ft3; do
  evalcfg "$base;D=$d;K=10;t=0.2;rule=obrs;step=D" "$csv" 0 --G_name "$G" --D_arch "$arch" --D_name "${DPATH[$d]}" --K 10 --save_range 01t --bg_thresh 0.2
done
bD=$(pick D D); echo "best D: $bD"
for K in 1 2 5 10 20 50; do
  evalcfg "$base;D=$bD;K=$K;t=0.2;rule=obrs;step=K" "$csv" 0 --G_name "$G" --D_arch "$arch" --D_name "${DPATH[$bD]}" --K $K --save_range 01t --bg_thresh 0.2
done
bK=$(pick K K); echo "best K: $bK"
for t in 0.12 0.16 0.2 0.25 0.3; do
  evalcfg "$base;D=$bD;K=$bK;t=$t;rule=obrs;step=t" "$csv" 0 --G_name "$G" --D_arch "$arch" --D_name "${DPATH[$bD]}" --K $bK --save_range 01t --bg_thresh $t
done
bt=$(pick t t); echo "best t: $bt"
if [ "$bK" != "1" ]; then
  evalcfg "$base;D=$bD;K=$bK;t=$bt;rule=threshold;step=rule" "$csv" 0 --G_name "$G" --D_arch "$arch" --D_name "${DPATH[$bD]}" --K $bK --save_range 01t --bg_thresh $bt --rule threshold
fi
for seed in 0 1 2; do
  evalcfg "$base;D=$bD;K=$bK;t=$bt;rule=obrs;step=final" "$csv" $seed --G_name "$G" --D_arch "$arch" --D_name "${DPATH[$bD]}" --K $bK --save_range 01t --bg_thresh $bt
done
