import os
import scanpy as sc
import pandas as pd
import anndata
import gc
import time
import numpy as np
import scipy
from sklearn.cluster import SpectralClustering
from collections import Counter
import warnings
import argparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, silhouette_score
from sklearn.metrics import jaccard_score
from banditpam import KMedoids
from sklearn.metrics.pairwise import pairwise_distances

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_silhouette_metric(alg_metric):
    if 'pearson' in alg_metric:
        return 'precomputed'
    elif 'cos' in alg_metric:
        return 'cosine'
    elif 'L1' in alg_metric:
        return 'manhattan'
    else:
        return 'euclidean'


def compute_score(data, key, rd_metric, alg_metric, target='label'):
    nmi = normalized_mutual_info_score(data.obs[target], data.obs[key])
    ari = adjusted_rand_score(data.obs[target], data.obs[key])
    fm = fowlkes_mallows_score(data.obs[target], data.obs[key])
    
    metric = get_silhouette_metric(alg_metric)
    
    if rd_metric == 'hvg':
        if metric == 'precomputed':
            distance_matrix = pairwise_distances(data.obsp[f'dm_{alg_metric}'], metric='correlation')
            silhouette_avg = silhouette_score(distance_matrix, data.obs[key], metric='precomputed')
        else:
            silhouette_avg = silhouette_score(data.obsp[f'dm_{alg_metric}'], data.obs[key], metric=metric)
    else:
        if metric == 'precomputed':
            distance_matrix = pairwise_distances(data.obsm[f'X_{rd_metric}'], metric='correlation')
            silhouette_avg = silhouette_score(distance_matrix, data.obs[key], metric='precomputed')
        else:
            silhouette_avg = silhouette_score(data.obsm[f'X_{rd_metric}'], data.obs[key], metric=metric)
    
    return nmi, ari, fm, silhouette_avg
    
def get_stats(data, file, algorithm, target, rd_metric, alg_metric, write_path):
    key = algorithm + '_' + rd_metric + '_' + alg_metric 
    nmi, ari, fm, silhouette_avg = compute_score(data, key, rd_metric, alg_metric, target)
    
    entry = {
        'dataset': file,
        'algorithm': algorithm,
        'rd_metric': rd_metric,
        'alg_metric': alg_metric,
        'NMI': nmi,
        'ARI': ari,
        'FM': fm,
        'Silhouette': silhouette_avg,
        '#k': len(np.unique(data.obs[key]))
    }
    return entry

def compute_KNN(data, rd_metric='pcoa_l2', alg_metric='l2'):
    if 'time' not in data.uns:
        data.uns['time'] = {}
    start_time = time.time()

    if rd_metric == 'hvg':
        X = data.X
        X = X.toarray() if scipy.sparse.issparse(X) else X
        data.X = X  # Assign dense matrix back to data.X
        sc.pp.neighbors(data, n_neighbors=10, use_rep='X', metric=alg_metric, key_added=f'KNN_{alg_metric}')
    else:
        X = data.obsm[f'X_{rd_metric}']
        X = X.toarray() if scipy.sparse.issparse(X) else X
        data.obsm[f'X_{rd_metric}'] = X  # Assign dense matrix back to data.obsm
        sc.pp.neighbors(data, n_neighbors=10, use_rep=f'X_{rd_metric}', metric=alg_metric, key_added=f'KNN_{alg_metric}')

    data.uns['time'][f'KNN_{rd_metric}_{alg_metric}'] = time.time() - start_time
    return data

def compute_spectral(data, n_clusters=5, rd_metric='pcoa_l2', alg_metric='l2'):
    if 'time' not in data.uns:
        data.uns['time'] = {}
    start_time = time.time()

    knn_key = f'KNN_{alg_metric}_connectivities'

    if knn_key in data.obsp:
        knn_connectivity = data.obsp[knn_key]
    else:
        raise KeyError(f"Key {knn_key} not found in data.obsp")

    if isinstance(data.X, np.ndarray):
        X = data.X
    else:
        X = data.X.toarray()

    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
    labels = spectral.fit_predict(knn_connectivity)
    data.obs[f'spectral_{rd_metric}_{alg_metric}'] = labels
    
    data.uns['time'][f'spectral_{rd_metric}_{alg_metric}'] = time.time() - start_time
    return data

def main(dataset_file, output_path, algorithm):
    print(f"Job started with dataset: {dataset_file}, output path: {output_path}, algorithm: {algorithm}")

    path = os.path.dirname(dataset_file)
    file = os.path.basename(dataset_file)
    dataset = file[:file.rfind('_')]
    
    print(f"Loading dataset from {dataset_file}...")
    data = anndata.read_h5ad(dataset_file)
    print(f"Dataset loaded: [{dataset}] {data.X.shape}")

    res = pd.DataFrame(columns=['dataset', 'algorithm', 'rd_metric', 'alg_metric', 'NMI', 'ARI', 'FM', 'Silhouette', '#k'])

    for rd_metric in ['hvg', 'pca', 'pcoa_l1', 'pcoa_l2', 'pcoa_cosine', 'pcoa_correlation', 'isomap_l1', 'isomap_l2', 'isomap_cosine', 'isomap_correlation', 'kpca_l2', 'kpca_cosine']:
        for alg_metric in ['l1','l2','cosine','correlation']:
            start_time = time.time()
                    
            data = compute_KNN(data, rd_metric, alg_metric)
            data = compute_spectral(data, len(Counter(data.obs['label'])), rd_metric, alg_metric)

            entry = get_stats(data, file, 'spectral', 'label' , rd_metric, alg_metric, 'write_path')
            res.loc[len(res.index)] = entry.values()
    
            print(f"RD: {rd_metric[:3]:>3}, AL: {alg_metric[:3]:>3} {time.time()-start_time:.2f} seconds")
            
            res.to_csv(os.path.join(output_path, f'clustering_{algorithm}_{dataset}_0820.csv'), index=False)

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
