#!/bin/bash
#SBATCH -p mzhang
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8Gb
#SBATCH --job-name=ScCluster
#SBATCH --output=/home/yenhsian/test/BanditPAM/results/clustering_%A_%a.out
#SBATCH --error=/home/yenhsian/test/BanditPAM/err/clustering_%A_%a.err
#SBATCH -t 24:00:00
#SBATCH --array=0-6%7  # Adjust this based on the number of datasets and the desired concurrency

# List of datasets
DATASETS=(
'GorillaMiddleTemporalGyrus_7K_t0.h5ad'
'HumanHeartCardiomyopathies_7K_t0.h5ad'
'HumanDevelopingImmuneSystem_7K_t0.h5ad'
'HumanBreastGlobal_7K_t0.h5ad'
'MouseEmbryonicEndothelium_7K_t0.h5ad'
'HumanPBMCs_7K_t0.h5ad'
'HumanEmbryonicLimbCell_7K_t0.h5ad'
)

# Get the dataset for this task
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate lab

# Run the Python script with arguments
python /home/yenhsian/test/BanditPAM/experiment/run_banditpam_DR.py \
    --dataset_file /projects/zhanglab/users/david/data/7K/$DATASET \
    --output_path /home/yenhsian/test/BanditPAM/results/0723 \
    --algorithm banditpam
