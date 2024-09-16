import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.metrics import jaccard_score

def compute_score(data, key):
    label_true, label_pred = data.obs['label'], data.obs[key]
    nmi = normalized_mutual_info_score(label_true, label_pred)
    ari = adjusted_rand_score(label_true, label_pred)
    fm = fowlkes_mallows_score(label_true, label_pred)
    return nmi, ari, fm

def get_stats(data, file_name, dataset, algorithm, rd_metric, alg_metric):
    key = algorithm + '_' + rd_metric + '_' + alg_metric 
    nmi, ari, fm = compute_score(data, key = key)
    
    entry = {
        'data': file_name,
        'dataset': dataset,
        'algorithm': algorithm,
        'rd_metric': rd_metric,
        'alg_metric': alg_metric,
        'NMI': nmi,
        'ARI': ari,
        'FM': fm,
        '#k': len(np.unique(data.obs[key]))
    }
    return entry

def sample_adata(adata, min_cells_per_type, target_sample_size):
    """
    Subsamples an AnnData object to ensure a minimum number of cells per cell type,
    and balances the remaining sample according to the inverse of cell type frequencies.
    
    Parameters:
    - adata: AnnData object containing scRNA-seq data.
    - min_cells_per_type: int, minimum number of cells to keep for each cell type.
    - target_sample_size: int, total number of cells to sample.
    
    Returns:
    - AnnData object containing the subsampled data.
    """
    # Calculate the number of cells per cell type
    cell_counts = adata.obs['label'].value_counts()
    
    # Ensure at least min_cells_per_type for each cell type
    initial_samples = []
    for cell_type, count in cell_counts.items():
        if count >= min_cells_per_type:
            cell_indices = adata.obs.index[adata.obs['label'] == cell_type]
            sampled_indices = np.random.choice(cell_indices, size=min_cells_per_type, replace=False)
            initial_samples.extend(sampled_indices)
    
    # Calculate weights for additional sampling
    weights = cell_counts / cell_counts.sum()  # Normalize weights
    
    # Create a list of indices eligible for additional sampling
    eligible_indices = adata.obs.index.difference(initial_samples)
    eligible_labels = adata.obs.loc[eligible_indices, 'label']

    # Normalize the weights for eligible indices only
    weights = weights[eligible_labels]
    weights /= weights.sum()  # Ensure the weights sum to 1
    
    # Sample additional cells according to the normalized inverse frequencies
    additional_sample_size = max(target_sample_size - len(initial_samples), 0)
    if additional_sample_size > 0 and not eligible_labels.empty:
        additional_samples = np.random.choice(eligible_indices, size=additional_sample_size, replace=False, p=weights.values)
    else:
        additional_samples = []
    
    # Combine the two lists of indices
    final_sample_indices = np.unique(np.concatenate((initial_samples, additional_samples)))
    
    # Subset the original AnnData object to the final sampled data
    adata_subsampled = adata[adata.obs.index.isin(final_sample_indices)].copy()
    
    return adata_subsampled

# Example usage
# adata_subsampled = sample_adata(adata, min_cells_per_type=200, target_sample_size=10000)
