#!/bin/bash
# Effect of the image conversion at saving time, alone and combined with OBRS.
#   raw: starter behaviour (tanh output saved as is, negative pixels -> black)
#   01 : (x+1)/2, the inverse of the training normalisation
# Only the saved PNGs change; D always sees G's output in [-1, 1].
# Results -> results_convert.csv (+ summary at the end of the .out file)
echo "Launching convert_study"

sbatch <<'EOT'
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mem=256G
#SBATCH --account=m25146
#SBATCH --job-name=convert_study
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err
source venv/bin/activate
export DATA="/projects/m25146/data/"

for range in raw 01; do
  for cfg in "baseline|--K 1" "OBRS K=2 D_ft1|--K 2 --D_name D_ft1.pth" "OBRS K=5 D_mlp_es|--K 5 --D_name D_mlp_es.pth"; do
    name="${cfg%%|*}"; opts="${cfg#*|}"
    for seed in 0 1 2; do
      label="$name | $range"
      echo "=== $label | seed $seed"
      rm -rf samples
      python generate.py --batch_size 256 --seed $seed --save_range $range $opts
      python evaluate.py --label "$label | seed $seed" --out results_convert.csv
    done
  done
done

python - <<'PY'
import csv, statistics as st
from collections import defaultdict
rows = list(csv.DictReader(open('results_convert.csv')))
groups = defaultdict(list)
for r in rows:
    groups[r['label'].rsplit(' | seed', 1)[0]].append(r)
cols = ['FID', 'precision', 'recall', 'class_TV', 'accept_rate']
print(f"{'config':30s}" + ''.join(f'{c:>18s}' for c in cols))
for name, rs in groups.items():
    line = f'{name:30s}'
    for c in cols:
        v = [float(r[c]) for r in rs if r.get(c) not in (None, '')]
        line += f'{st.mean(v):11.4g} ± {st.pstdev(v):5.2g}' if v else f'{"-":>18s}'
    print(line)
PY
EOT
