#!/bin/bash
name="DataLab_A2_train"
outdir="runs/slurm_logs"
n_gpu=1
export DATA="/projects/m25146/data/"

    echo "Launching job for $name"
    
    sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet 
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:${n_gpu}
#SBATCH --time=00:20:00
#SBATCH --mem=10G
#SBATCH --account=m25146        # Your project account 
#SBATCH --job-name=gan_evaluate      # Job name
#SBATCH --output=${outdir}/%x_%j.out  # Standard output and error log
#SBATCH --error=${outdir}/%x_%j.err  # Error log
source .venv/bin/activate
# Run your evaluation script
python evaluate.py
EOT