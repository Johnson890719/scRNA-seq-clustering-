import time
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from collections import Counter

def load_data(file, verbose = True):
    if verbose:
        print('load: ',file)

    adata = anndata.read_h5ad(file)
    
    sc.pp.calculate_qc_metrics(adata, inplace=True, log1p=True)

    start_time = time.time()
    if verbose:
        print('n_obs × n_vars: ', adata.shape)
        print('running time: ', time.time() - start_time)
    return adata


def filter_top_k_clusters(adata, target, k, verbose=True):
    """
    Filters the AnnData object to retain only samples belonging to the top `k` clusters based on their size.
    
    Parameters:
        adata (AnnData): The annotated data matrix.
        target (str): The column in `adata.obs` that contains the cluster labels.
        k (int): The number of top clusters to retain.
    
    Returns:
        AnnData: A filtered AnnData object containing only the top `k` clusters.
    """
    # Ensure that 'nan' values are handled; adjust based on actual data type (str 'nan' or np.nan)
    adata = adata[adata.obs[target].notna()]
    adata = adata[adata.obs[target] != 'nan']

    # Get a sorted list of labels based on the number of items, in descending order
    counter = Counter(adata.obs[target])
    labels_items = sorted(counter.items(), key=lambda x: -x[1])
    
    # Extract the labels of the top k clusters
    top_k_labels = [item[0] for item in labels_items[:k]]

    # Filter the data to only include top k clusters
    adata_filtered = adata[adata.obs[target].isin(top_k_labels)]

    # Update counter to reflect the filtered data
    counter_filtered = Counter(adata_filtered.obs[target])

    if verbose:
        print('#k:', len(counter_filtered))
    
    return adata_filtered, counter_filtered

def label_encode(adata, target):
    labels, labels_dict = [], {}
    cur = 0
    for c in pd.Series(adata.obs[target]):
        if c not in labels_dict:
            labels_dict[c] = cur
            cur += 1 
        labels.append(labels_dict[c])

    adata.obs['label'] = labels
    return labels_dict
