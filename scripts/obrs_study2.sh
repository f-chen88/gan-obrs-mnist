#!/bin/bash
# OBRS study, part 2: fixing the density-ratio estimate.
# Part 1 showed that the 10-epoch fine-tuned D is over-confident (E_fake[r] ~ 0.2-0.4)
# and that OBRS then over-selects easy digits (1s), hurting FID.
# Here: 4 discriminators (end of training, fine-tuned 1 / 3 / 10 epochs)
#       x {raw ratio, Platt-calibrated ratio}  x  K in {1.5, 2, 3}  x  3 seeds.
# Results -> results_obrs2.csv (+ summary at the end of the .out file).
echo "Launching obrs_study2"

sbatch <<'EOT'
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=256G
#SBATCH --account=m25146
#SBATCH --job-name=obrs_study2
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err
source venv/bin/activate
export DATA="/projects/m25146/data/"

# Shorter fine-tunings of D (D_ft.pth = 10 epochs already exists from part 1)
python finetune_D.py --epochs 1 --lr 0.0001 --G_name G.pth --out_name D_ft1.pth
python finetune_D.py --epochs 3 --lr 0.0001 --G_name G.pth --out_name D_ft3.pth

for D in D.pth D_ft1.pth D_ft3.pth D_ft.pth; do
  python calibrate_D.py --D_name $D
done

for D in D.pth D_ft1.pth D_ft3.pth D_ft.pth; do
  calib="checkpoints/${D%.pth}_calib.json"
  for mode in raw cal; do
    opt=""; [ "$mode" = cal ] && opt="--calib $calib"
    for K in 1.5 2 3; do
      for seed in 0 1 2; do
        label="$D $mode K=$K"
        echo "=== $label | seed $seed"
        rm -rf samples
        python generate.py --batch_size 256 --seed $seed --K $K --D_name $D $opt
        python evaluate.py --label "$label | seed $seed" --out results_obrs2.csv
      done
    done
  done
done

python - <<'PY'
import csv, statistics as st
from collections import defaultdict
rows = list(csv.DictReader(open('results_obrs2.csv')))
groups = defaultdict(list)
for r in rows:
    groups[r['label'].rsplit(' | seed', 1)[0]].append(r)
cols = ['FID', 'precision', 'recall', 'class_TV', 'mean_r']
print('Reference (part 1): no rejection K=1 -> FID 59.75, P 0.692, R 0.809, class_TV 0.144')
print(f"{'config':26s}" + ''.join(f'{c:>18s}' for c in cols))
for name, rs in sorted(groups.items(), key=lambda kv: st.mean(float(r['FID']) for r in kv[1])):
    line = f'{name:26s}'
    for c in cols:
        v = [float(r[c]) for r in rs if r.get(c) not in (None, '')]
        line += f'{st.mean(v):11.4g} ± {st.pstdev(v):5.2g}' if v else f'{"-":>18s}'
    print(line)
PY
EOT
