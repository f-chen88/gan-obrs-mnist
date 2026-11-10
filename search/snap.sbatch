#!/bin/bash
#SBATCH -p mesonet
#SBATCH --account=m25146
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --job-name=s_snap
#SBATCH --output=outputs/search_%x_%A_%a.out
# Stage 2: for every snapshot of one run: fine-tune its D for 1 epoch against the frozen G,
# then evaluate at t = 0.2 without rejection (K=1) and with OBRS K=10 (seed 0).
source "$ROOT/search/common.sh"
name=$(run_name "$SLURM_ARRAY_TASK_ID")
arch=$(run_arch "$name")
csv="$ROOT/$RES/snap_$name.csv"
mkdir -p "$ROOT/$RES"
workdir "snap_$name"
for ep in 25 50 75 100 125 150 175 200 225 250 275 300; do
  G="search/$name/G_ep$ep.pth"; D0="search/$name/D_ep$ep.pth"; D1="search/$name/Dft1_ep$ep.pth"
  [ -f "checkpoints/$G" ] || { echo "missing $G"; continue; }
  [ -f "checkpoints/$D1" ] || python finetune_D.py --D_arch "$arch" --epochs 1 --G_name "$G" --D_init "$D0" --out_name "$D1"
  evalcfg "run=$name;ep=$ep;D=ft1;K=1;t=0.2;rule=obrs;step=snap"  "$csv" 0 --G_name "$G" --K 1 --save_range 01t --bg_thresh 0.2
  evalcfg "run=$name;ep=$ep;D=ft1;K=10;t=0.2;rule=obrs;step=snap" "$csv" 0 --G_name "$G" --D_arch "$arch" --D_name "$D1" --K 10 --save_range 01t --bg_thresh 0.2
done
