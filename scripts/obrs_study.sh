#!/bin/bash
# OBRS study on the trained baseline (checkpoints/G.pth, kept frozen).
#   0. Fine-tune D against the frozen G   -> checkpoints/D_ft.pth
#   1. Budget sweep: K = 1 ... 10, 3 seeds, full 10k evaluation batch each time
#   2. Ablation "rule": optimal OBRS vs hard threshold (keep top 1/K by r), same budget
#   3. Ablation "ratio estimator": D fine-tuned vs D from the end of GAN training
# Every run adds one row to results.csv with FID, P, R, density, coverage,
# class balance, acceptance rate, total generation time.
echo "Launching obrs_study"

sbatch <<'EOT'
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=256G
#SBATCH --account=m25146
#SBATCH --job-name=obrs_study
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err
source venv/bin/activate
export DATA="/projects/m25146/data/"

run () {   # run "<label>" <generate.py options...>
  label="$1"; shift
  for seed in 0 1 2; do
    echo "=== $label | seed $seed"
    rm -rf samples
    python generate.py --batch_size 256 --seed $seed "$@"
    python evaluate.py --label "$label | seed $seed" --out results_obrs.csv
  done
}

# 0. Ratio estimator matched to the final generator
python finetune_D.py --epochs 10 --lr 0.0001 --G_name G.pth --out_name D_ft.pth

# 1. Budget sweep (K = 1 is the baseline: no rejection)
for K in 1 1.25 1.5 2 3 5 10; do
  run "OBRS K=$K" --K $K --rule obrs --D_name D_ft.pth
done

# 2. Same budget, deterministic threshold instead of the optimal rule
for K in 2 5; do
  run "threshold K=$K" --K $K --rule threshold --D_name D_ft.pth
done

# 3. Same rule, ratio estimated with the D saved at the end of training
for K in 2 5; do
  run "OBRS K=$K, D not fine-tuned" --K $K --rule obrs --D_name D.pth
done

# Summary: mean +- std over the 3 seeds
python - <<'PY'
import csv, statistics as st
from collections import defaultdict
rows = list(csv.DictReader(open('results_obrs.csv')))
groups = defaultdict(list)
for r in rows:
    groups[r['label'].rsplit(' | seed', 1)[0]].append(r)
cols = ['FID', 'precision', 'recall', 'class_TV', 'accept_rate', 'gen_time_s']
print(f"{'config':32s}" + ''.join(f'{c:>18s}' for c in cols))
for name, rs in groups.items():
    line = f'{name:32s}'
    for c in cols:
        v = [float(r[c]) for r in rs if r.get(c) not in (None, '')]
        line += f'{st.mean(v):11.4g} ± {st.pstdev(v):5.2g}' if v else f'{"-":>18s}'
    print(line)
PY
EOT
