#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1            # we manage parallelism manually
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --account=Soltoggio2025a
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "=========================================="

module purge
module load CUDA/12.4.0
source ~/.bashrc
conda activate lora_retriever

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

bash eval_all.sh

echo "=========================================="
echo "All runs completed at: $(date)"
echo "=========================================="
