import os
import scanpy as sc
import pandas as pd
import anndata
import gc
import time
import numpy as np
from sklearn.cluster import SpectralClustering
from pipeline import dl, pp, cl, pl, utils
from collections import Counter
import warnings
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)

def compute_spectral(data, n_clusters=5, rd_metric='pcoa_l2', alg_metric='l2'):
    ...
    return data

def main(dataset_file, output_path, algorithm):
    print(f"Job started with dataset: {dataset_file}, output path: {output_path}, algorithm: {algorithm}")

    path = os.path.dirname(dataset_file)
    file = os.path.basename(dataset_file)
    dataset = file[:file.rfind('_')]
    
    print(f"Loading dataset from {dataset_file}...")
    data = anndata.read_h5ad(dataset_file)
    print(f"Dataset loaded: [{dataset}] {data.X.shape}")

    columns = ['data', 'dataset', 'algorithm', 'rd_metric', 'alg_metric', 'NMI', 'ARI', 'FM', '#k']
    res = pd.DataFrame(columns=columns)

    for rd_metric in ['pca', 'pcoa_l1','pcoa_l2','pcoa_cosine','pcoa_correlation']:
        for alg_metric in ['l1','l2','cosine','correlation']:
            start_time = time.time()
                    
            data = cl.compute_KNN(data, rd_metric, alg_metric)
            data = compute_spectral(data, 20, rd_metric, alg_metric)

            entry = utils.get_stats(data, file, dataset, algorithm, rd_metric, alg_metric)
            res.loc[len(res.index)] = entry.values()
    
            print(f"RD: {rd_metric[:3]:>3}, AL: {alg_metric[:3]:>3} {time.time()-start_time:.2f} seconds")
            
            res.to_csv(os.path.join(output_path, f'clustering_{algorithm}_{file}.csv'), index=False)

    data = None
    gc.collect()
    print("Job completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run clustering experiment')
    parser.add_argument('--dataset_file', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output files')
    parser.add_argument('--algorithm', type=str, required=True, help='Algorithm name (e.g., spectral)')
    args = parser.parse_args()

    main(args.dataset_file, args.output_path, args.algorithm)
