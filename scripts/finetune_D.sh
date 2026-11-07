#!/bin/bash
name="finetune_D"
outdir="outputs"
export DATA="/projects/m25146/data/"
    echo "Launching $name"

    sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=256G
#SBATCH --account=m25146
#SBATCH --job-name=finetune_D
#SBATCH --output=${outdir}/%x_%j.out
#SBATCH --error=${outdir}/%x_%j.err
source venv/bin/activate
python finetune_D.py --epochs 10 --lr 0.0001
EOT
