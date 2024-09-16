#!/bin/bash
#SBATCH -p mzhang
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12Gb
#SBATCH --job-name=ScClusterMetric
#SBATCH --output=/home/chunchua/ScClusterMetric/experiments/output/out/isomap_10K_%A_%a.out
#SBATCH --error=/home/chunchua/ScClusterMetric/experiments/output/err/isomap_10K_%A_%a.err
#SBATCH -t 2:00:00
#SBATCH --array=0-29%10  # Adjust this based on the number of datasets and the desired concurrency


# List of datasets in the specified size category
DATASETS=(
"Nathan_10K_t1.h5ad"
"TMSdroplet_10K_t2.h5ad"
"MouseRetina_10K_t1.h5ad"
"MouseKidneyAtlas_10K_t1.h5ad"
"AdultHumanKidney_10K_t2.h5ad"
"LiverAtlas_10K_t1.h5ad"
"TS10X_10K_t1.h5ad"
"TSsmartseq2_10K_t2.h5ad"
"MouseKidneyAtlas_10K_t0.h5ad"
"LiverAtlas_10K_t0.h5ad"
"TS10X_10K_t0.h5ad"
"Cano_10K_t2.h5ad"
"Nathan_10K_t0.h5ad"
"TMSfacs_10K_t2.h5ad"
"MouseRetina_10K_t0.h5ad"
"MouseKidneyAtlas_10K_t2.h5ad"
"AdultHumanKidney_10K_t1.h5ad"
"LiverAtlas_10K_t2.h5ad"
"TS10X_10K_t2.h5ad"
"TSsmartseq2_10K_t1.h5ad"
"Cano_10K_t0.h5ad"
"Nathan_10K_t2.h5ad"
"TMSdroplet_10K_t1.h5ad"
"MouseRetina_10K_t2.h5ad"
"TMSfacs_10K_t0.h5ad"
"TMSdroplet_10K_t0.h5ad"
"TMSfacs_10K_t1.h5ad"
"AdultHumanKidney_10K_t0.h5ad"
"TSsmartseq2_10K_t0.h5ad"
"Cano_10K_t1.h5ad"
)


# Get the dataset for this task
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate lab


# Run the Python script with arguments
python /home/chunchua/ScClusterMetric/experiments/run_isomap.py \
    --dataset_file /projects/zhanglab/users/david/data/10K/$DATASET \
    --output_path /home/chunchua/ScClusterMetric/experiments/output
