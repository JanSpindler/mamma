#!/bin/bash
#SBATCH --job-name=mamma_thuman
#SBATCH --array=0-11
#SBATCH --partition=mlgpu_long      # A40; switch to mlgpu_medium if runs are < 24h
#SBATCH --account=ag_ifi_klein
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SUBJECTS=(
  2026_05_19_anchor_017
  2026_05_19_anchor_018
  2026_05_19_anchor_019
  2026_05_22_anchor_020
  2026_05_22_anchor_021
  2026_05_22_anchor_022
  2026_05_22_anchor_023
  2026_05_22_anchor_024
  2026_05_22_anchor_025
  2026_05_22_anchor_026
  2026_05_22_anchor_027
  2026_05_22_anchor_028
)

PROC_ROOT=/lustre/mlnvme/data/jspindle_hpc-anchor/processed
SUBJECT_DIR="$PROC_ROOT/${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

# Fail early if the subject data is missing (saves a wasted GPU allocation).
# Layout is flattened (fix_images_thuman.py): cameras live directly under the
# subject dir, e.g. <subject>/<cam>/..., with no images/ wrapper.
[[ -f "$SUBJECT_DIR/calibration.json" ]] || { echo "MISSING calibration.json: $SUBJECT_DIR"; exit 1; }
shopt -s nullglob
cam_dirs=("$SUBJECT_DIR"/*/)
[[ ${#cam_dirs[@]} -gt 0 ]] || { echo "NO camera subdirs in: $SUBJECT_DIR"; exit 1; }

module load CUDA/12.8.0                   # confirm exact name: module spider CUDA
eval "$(conda shell.bash hook)"
conda activate mamma

cd /home/jspindle_hpc/Code/Comparison/mamma   # relative --cfg path needs this

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID subject=$(basename "$SUBJECT_DIR") node=$(hostname)"
bash run_anchor.sh 0 "$SUBJECT_DIR"
echo "[$(date)] DONE task=$SLURM_ARRAY_TASK_ID"