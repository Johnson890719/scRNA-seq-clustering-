#!/bin/bash
#SBATCH -p mzhang
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12Gb
#SBATCH --job-name=ScClusterMetric
#SBATCH --output=/home/chunchua/ScClusterMetric/experiments/output/out/kpca_%A_%a.out
#SBATCH --error=/home/chunchua/ScClusterMetric/experiments/output/err/kpca_%A_%a.err
#SBATCH -t 2:00:00
#SBATCH --array=0-29%10  # Adjust this based on the number of datasets and the desired concurrency


# List of datasets in the specified size category
DATASETS=(
"TMSfacs_5K_t0.h5ad"
"Cano_5K_t2.h5ad"
"TSsmartseq2_5K_t1.h5ad"
"TMSdroplet_5K_t1.h5ad"
"Nathan_5K_t2.h5ad"
"MouseKidneyAtlas_5K_t2.h5ad"
"LiverAtlas_5K_t0.h5ad"
"TMSdroplet_5K_t0.h5ad"
"LiverAtlas_5K_t1.h5ad"
"MouseRetina_5K_t2.h5ad"
"AdultHumanKidney_5K_t2.h5ad"
"TMSfacs_5K_t1.h5ad"
"TS10X_5K_t2.h5ad"
"TSsmartseq2_5K_t0.h5ad"
"TMSdroplet_5K_t2.h5ad"
"Nathan_5K_t1.h5ad"
"MouseKidneyAtlas_5K_t1.h5ad"
"AdultHumanKidney_5K_t0.h5ad"
"MouseRetina_5K_t0.h5ad"
"Cano_5K_t1.h5ad"
"TS10X_5K_t0.h5ad"
"TSsmartseq2_5K_t2.h5ad"
"TMSfacs_5K_t2.h5ad"
"Cano_5K_t0.h5ad"
"TS10X_5K_t1.h5ad"
"Nathan_5K_t0.h5ad"
"MouseKidneyAtlas_5K_t0.h5ad"
"LiverAtlas_5K_t2.h5ad"
"MouseRetina_5K_t1.h5ad"
"AdultHumanKidney_5K_t1.h5ad"
)


# Get the dataset for this task
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate lab


# Run the Python script with arguments
python /home/chunchua/ScClusterMetric/experiments/run_kpca.py \
    --dataset_file /projects/zhanglab/users/david/data/5K/$DATASET \
    --output_path /home/chunchua/ScClusterMetric/experiments/output
