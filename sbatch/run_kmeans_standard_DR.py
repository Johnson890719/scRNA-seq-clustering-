import os
import scanpy as sc
import pandas as pd
import scipy
import anndata
import gc
import time
import numpy as np
from sklearn.cluster import SpectralClustering
from collections import Counter
import warnings
import argparse
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, silhouette_score
from sklearn.metrics import jaccard_score
from nltk.cluster import KMeansClusterer, cosine_distance
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
    alg_metric_map = {
    'L1': 'l1',
    'L2': 'l2',
    'cos': 'cosine',
    'pearson': 'correlation'
    }
    alg_metric = alg_metric_map.get(alg_metric, alg_metric)
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
    
def l1_distance(x, y):
    return np.sum(np.abs(x - y))

def l2_distance(x, y):
    return np.linalg.norm(x - y)

def cosine_distance(x, y):
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return 1 - (dot_product / (norm_x * norm_y))

def pearson_distance(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    denom = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return 1 - num / denom

def kmeans_plusplus_initialization(data, k, distance_metric):
    """
    Initialize centroids using the K-means++ algorithm.

    Parameters:
    - data: numpy array of data points (shape: [n_samples, n_features]).
    - k: number of clusters.
    - distance_metric: function to compute the distance between two points.

    Returns:
    - initial_means: list of k initial centroids.
    """
    n_samples, _ = data.shape
    # Randomly choose the first centroid.
    centroids_idx = [np.random.randint(0, n_samples)]
    centroids = [data[centroids_idx[0]]]

    for _ in range(1, k):
        distances = np.array([min([distance_metric(x, centroid) for centroid in centroids]) for x in data])
        probabilities = distances ** 2
        probabilities /= probabilities.sum()
        next_centroid_idx = np.random.choice(n_samples, p=probabilities)
        centroids.append(data[next_centroid_idx])

    return centroids

# Distance metric dictionary
DISTANCE_METRICS = {
    'L1': l1_distance,
    'L2': l2_distance,
    'cosine': cosine_distance,
    'pearson': pearson_distance}

def compute_kmeans(data, k, key, rd_metric, alg_metric, k_plus_plus = True):
    if 'time' not in data.uns:
        data.uns['time'] = {}
    start_time = time.time()

    if rd_metric == 'hvg':
        X = data.X
    else:
        X = data.obsm[f'X_{rd_metric}']
        
    X = X.toarray() if scipy.sparse.issparse(X) else X

    distance_function = DISTANCE_METRICS[alg_metric]

    if k_plus_plus == True:
        # Use K-means++ for initial centroid selection
        initial_means = kmeans_plusplus_initialization(X, k, distance_function)
    
        # Perform K-means clustering
        clusterer = KMeansClusterer(k, distance_function, initial_means=initial_means, avoid_empty_clusters = True)
        clusters = clusterer.cluster(X, True)
        data.obs[key] = clusters
    else: 
        clusterer = KMeansClusterer(k, distance_function, avoid_empty_clusters = True)
        clusters = clusterer.cluster(X, True)
        data.obs[key] = clusters
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
        for alg_metric in ['L1','L2','cosine','pearson']:
            start_time = time.time()
                    
            data = compute_kmeans(data, len(Counter(data.obs['label'])), 'kmeans_' + rd_metric + '_' + alg_metric , rd_metric, alg_metric, k_plus_plus = False)

            entry = get_stats(data, file, 'kmeans', 'label' , rd_metric, alg_metric, 'write_path')
            res.loc[len(res.index)] = entry.values()
    
            print(f"RD: {rd_metric[:3]:>3}, AL: {alg_metric[:3]:>3} {time.time()-start_time:.2f} seconds")
            
            res.to_csv(os.path.join(output_path, f'clustering_{algorithm}_{file}_0820.csv'), index=False)

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
