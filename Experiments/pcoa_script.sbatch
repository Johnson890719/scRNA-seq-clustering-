#!/bin/bash
#SBATCH -p mzhang
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64Gb
#SBATCH --job-name=ScClusterMetric
#SBATCH --output=/home/chunchua/ScClusterMetric/experiments/output/out/pcoa_%A_%a.out
#SBATCH --error=/home/chunchua/ScClusterMetric/experiments/output/err/pcoa_%A_%a.err
#SBATCH -t 1:00:00
#SBATCH --array=0-23%5  # Adjust this based on the number of datasets and the desired concurrency


# List of datasets in the specified size category
DATASETS=(
    "Cano_7K_t2.h5ad"
    "LiverAtlas_7K_t1.h5ad"
    "Nathan_7K_t2.h5ad"
    "TSsmartseq2_7K_t1.h5ad"
    "TS10X_7K_t2.h5ad"
)


# Get the dataset for this task
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate lab


# Run the Python script with arguments
python /home/chunchua/ScClusterMetric/experiments/run_pcoa.py \
    --dataset_file /projects/zhanglab/users/david/data/7K/$DATASET \
    --output_path /home/chunchua/ScClusterMetric/experiments/output
