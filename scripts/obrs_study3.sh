#!/bin/bash
# OBRS study, part 3: does the ARCHITECTURE of the ratio estimator D limit the precision gain?
# G frozen (checkpoints/G.pth). Three estimators:
#   D_ft1.pth    MLP, fine-tuned 1 epoch from D.pth (current best, part 2)       [reference]
#   D_mlp_es.pth MLP, from D.pth, early stopping on a validation split            [same protocol as CNN]
#   D_cnn_es.pth CNN, from scratch, early stopping on a validation split
# x K in {1.5, 2, 3, 5} x 3 seeds, plus the K=1 baseline. Results -> results_obrs3.csv
echo "Launching obrs_study3"

sbatch <<'EOT'
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=256G
#SBATCH --account=m25146
#SBATCH --job-name=obrs_study3
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err
source venv/bin/activate
export DATA="/projects/m25146/data/"

python train_ratio_D.py --arch mlp --init D.pth --out D_mlp_es.pth
python train_ratio_D.py --arch cnn --out D_cnn_es.pth

for seed in 0 1 2; do
  echo "=== baseline K=1 | seed $seed"
  rm -rf samples
  python generate.py --batch_size 256 --seed $seed --K 1
  python evaluate.py --label "baseline K=1 | seed $seed" --out results_obrs3.csv
done

for cfg in "mlp D_ft1.pth" "mlp D_mlp_es.pth" "cnn D_cnn_es.pth"; do
  set -- $cfg; arch=$1; D=$2
  for K in 1.5 2 3 5; do
    for seed in 0 1 2; do
      label="$D K=$K"
      echo "=== $label | seed $seed"
      rm -rf samples
      python generate.py --batch_size 256 --seed $seed --K $K --D_arch $arch --D_name $D
      python evaluate.py --label "$label | seed $seed" --out results_obrs3.csv
    done
  done
  python obrs_grid.py --K 2 --D_arch $arch --D_name $D --prefix "${D%.pth}_K2"
done

python - <<'PY'
import csv, json, statistics as st
from collections import defaultdict
for d in ['D_mlp_es', 'D_cnn_es']:
    t = json.load(open(f'checkpoints/{d}_trainlog.json'))
    print(f"{d}: arch={t['arch']}, params={t['params']}, best epoch={t['best_epoch']:.2f}, "
          f"val BCE={t['val_bce']:.4f}, val acc={t['val_acc']:.3f}, mean r={t['mean_r']:.3f}")
rows = list(csv.DictReader(open('results_obrs3.csv')))
groups = defaultdict(list)
for r in rows:
    groups[r['label'].rsplit(' | seed', 1)[0]].append(r)
cols = ['FID', 'precision', 'recall', 'class_TV', 'accept_rate']
print(f"{'config':24s}" + ''.join(f'{c:>18s}' for c in cols))
for name, rs in groups.items():
    line = f'{name:24s}'
    for c in cols:
        v = [float(r[c]) for r in rs if r.get(c) not in (None, '')]
        line += f'{st.mean(v):11.4g} ± {st.pstdev(v):5.2g}' if v else f'{"-":>18s}'
    print(line)
PY
EOT
