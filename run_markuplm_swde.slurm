#!/bin/bash
#SBATCH --job-name=markuplm_swde
#SBATCH --output=logs/markuplm_swde_%j.out
#SBATCH --error=logs/markuplm_swde_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -A hcg@v100
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread

# ── Environment ──────────────────────────────────────────────────────────────
cd $SCRATCH

module purge
module load python/3.9.12

source venv_markuplm/bin/activate

export HF_HOME="$SCRATCH/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

# ── Run ──────────────────────────────────────────────────────────────────────
mkdir -p logs

python3 "MarkupLM Finetuning/markuplm_finetune_swde.py" \
    --swde_path      "MarkupLM Finetuning/SWDE" \
    --processed_path "MarkupLM Finetuning/SWDE/processed" \
    --output_dir     "MarkupLM Finetuning/results_markuplm" \
    --model_name_or_path "$SCRATCH/markuplm-base-local" \
    --verticals      auto book camera job movie nbaplayer restaurant university \
    --cuda           0
