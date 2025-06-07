#!/bin/bash
#SBATCH -p mzhang
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8Gb
#SBATCH --job-name=ScCluster
#SBATCH --output=/home/yenhsian/test/BanditPAM/results/clustering_%A_%a.out
#SBATCH --error=/home/yenhsian/test/BanditPAM/err/clustering_%A_%a.err
#SBATCH -t 24:00:00
#SBATCH --array=0-9%10  # Adjust this based on the number of datasets and the desired concurrency

# List of datasets
DATASETS=(
"TS10X.h5ad"
"GSM2230759_human3_umifm_counts.h5ad"
"GSM2230760_human4_umifm_counts.h5ad"
"GSM2230761_mouse1_umifm_counts.h5ad"
"GSM2230758_human2_umifm_counts.h5ad"
"TMSfacs.h5ad"
"GSM2230762_mouse2_umifm_counts.h5ad"
"TMSdroplet.h5ad"
"GSM2230757_human1_umifm_counts.h5ad"
"TSsmartseq2.h5ad"
)

# Get the dataset for this task
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate lab

# Run the Python script with arguments
python /home/yenhsian/test/BanditPAM/experiment/run_kmeans_standard_DR.py \
    --dataset_file /projects/zhanglab/users/johnson/gold+silver/$DATASET \
    --output_path /home/yenhsian/test/BanditPAM/results/0829_comparable \
    --algorithm kmeans_standard
