import time
import numpy as np
import scanpy as sc
from scipy.spatial.distance import squareform, pdist
from sklearn.mixture import GaussianMixture
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap



def distance_matrix(data, metric='l2'):
    """
    Calculate a distance matrix from the given data using the specified metric.

    Parameters:
        data (anndata.AnnData): An AnnData object containing the data matrix.
        metric (str): The distance metric to use ('l1', 'l2', 'cosine', or 'correlation').

    Returns:
        numpy.ndarray: A symmetric matrix of pairwise distances.

    Raises:
        ValueError: If an invalid distance metric is specified.
    """
    # Translate metric to scipy-compatible names using a dictionary
    metric_dict = {
        'l1': 'cityblock',
        'l2': 'euclidean',
        'cosine': 'cosine',
        'correlation': 'correlation'
    }
    
    if metric in metric_dict:
        scipy_metric = metric_dict[metric]
    else:
        valid_options = ', '.join(f"'{m}'" for m in metric_dict.keys())
        raise ValueError(f"Invalid metric '{metric}'. Valid options are {valid_options}.")

    # Convert sparse matrix to dense if necessary
    data_array = data.X.toarray() if hasattr(data, 'X') and hasattr(data.X, "toarray") else data.X

    # Compute the pairwise distance matrix
    return squareform(pdist(data_array, metric=scipy_metric))

def compute_distance_matrix(data, metric, verbose = False):
    
    if 'time' not in data.uns:
        data.uns['time'] = {}
        
    start_time = time.time()
    dm = distance_matrix(data, metric)

    # write
    data.obsp[f'dm_{metric}'] = dm.astype('float32')
    data.uns['time'][f'dm_{metric}'] = time.time() - start_time # running time

    if verbose: 
        print(f"dm: {dm.shape}, running time: {time.time() - start_time}")

    return data

def pcoa_full(dm, n_components=50):
    """
    Perform Principal Coordinates Analysis (PCoA) on a given distance matrix.

    Principal Coordinates Analysis (also known as Metric Multidimensional Scaling) is a
    method that aims to position n points in a p-dimensional space such that the distances
    between these points are preserved as well as possible. The input for PCoA is a distance
    matrix between samples, and it outputs the coordinates of each sample in the reduced
    dimensional space.

    Parameters:
        dm (numpy.ndarray): A square symmetric distance matrix (n x n) where n is the number of samples.
        n_components (int): The number of components (dimensions) to retain in the output.

    Returns:
        numpy.ndarray: The coordinates of each sample in the reduced space (n x n_components).
        Each row corresponds to a sample, with its coordinates in the reduced space.

    Steps:
        1. Center the distance matrix to make the matrix double-centered.
        2. Compute the eigenvalue decomposition of the centered matrix.
        3. Sort the eigenvalues and corresponding eigenvectors in descending order.
        4. Select the top `n_components` eigenvectors.
        5. Scale the eigenvectors by the square root of their corresponding eigenvalues to obtain the coordinates.

    Note:
        - The eigenvalue decomposition is computed using `numpy.linalg.eigh`, which is optimized
          for symmetric matrices (like the one obtained after centering the distance matrix).
    """
    
    # Step 1: Center the distance matrix
    n = dm.shape[0]
    H = np.eye(n, dtype='float32') - np.ones((n, n), dtype='float32') / n  # Centering matrix
    B = -0.5 * H.dot(dm**2).dot(H)  # Double centered matrix B

    # Step 2: Compute the eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(B)  # Using 'eigh' as B is symmetric
    
    # Step 3: Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Step 4: Select the top n_components
    eigvals = eigvals[:n_components]
    eigvecs = eigvecs[:, :n_components]

    # Step 5: Scale the eigenvectors by the square root of their corresponding eigenvalues
    coordinates = eigvecs * np.sqrt(np.abs(eigvals))
    
    return coordinates, eigvals

def compute_pcoa_full(data, metric = 'l2', n_components = 50, verbose = False):
    """
    Computes Principal Coordinates Analysis (PCoA) on the specified distance matrix within
    an AnnData object, updates the object with results, and optionally prints details.

    Parameters:
        adata (AnnData): The annotated data matrix to perform PCoA on.
        n_components (int): Number of principal coordinates to compute.
        metric (str): The metric used to create the distance matrix stored in adata.obsp.
        verbose (bool): If True, print computation time and results summary.

    Returns:
        AnnData: The updated AnnData object with PCoA results added.
    """
    if 'time' not in data.uns:
        data.uns['time'] = {}
        
    start_time = time.time()
    coordinates, eigvals = pcoa(data.obsp[f'dm_{metric}'], n_components)

    # write
    data.obsm[f'X_{metric}'] = coordinates
    data.uns['pcoa'] = {metric + '_eigvals': eigvals}
    data.uns['time'][f'pcoa_{metric}'] = time.time() - start_time

    if verbose:
        print(f'pcoa: {coordinates.shape}, running time: {time.time() - start_time}')

    return data

def pcoa(dm, n_components=50):
    # Step 1: Center the distance matrix
    n = dm.shape[0]
    H = np.eye(n, dtype='float32') - np.ones((n, n), dtype='float32') / n  # Centering matrix
    B = -0.5 * H.dot(dm**2).dot(H)  # Double centered matrix B

    # Step 2: Compute the top k eigenvalue decomposition
    # eigsh finds the k largest eigenvalues and eigenvectors for symmetric matrices
    eigvals, eigvecs = eigsh(B, k=n_components, which='LA')  # 'LA' -> Largest Algebraic

    # The eigsh function returns the eigenvalues, eigenvectors in ascending order
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    # Step 3: Scale the eigenvectors by the square root of their corresponding eigenvalues
    coordinates = eigvecs * np.sqrt(np.abs(eigvals))
    
    return coordinates, eigvals

def compute_pcoa(data, metric = 'l2', n_components = 50, verbose = False):
    """
    Computes Principal Coordinates Analysis (PCoA) on the specified distance matrix within
    an AnnData object, updates the object with results, and optionally prints details.

    Parameters:
        adata (AnnData): The annotated data matrix to perform PCoA on.
        n_components (int): Number of principal coordinates to compute.
        metric (str): The metric used to create the distance matrix stored in adata.obsp.
        verbose (bool): If True, print computation time and results summary.

    Returns:
        AnnData: The updated AnnData object with PCoA results added.
    """
    if 'time' not in data.uns:
        data.uns['time'] = {}
        
    start_time = time.time()
    coordinates, eigvals = pcoa(data.obsp[f'dm_{metric}'], n_components)

    # write
    data.obsm[f'X_pcoa_{metric}'] = coordinates
    data.uns['pcoa'] = {metric + '_eigvals': eigvals}
    data.uns['time'][f'pcoa_{metric}'] = time.time() - start_time

    if verbose:
        print(f'pcoa: {coordinates.shape}, running time: {time.time() - start_time}')

    return data

def compute_kernelPCA(data, kernel='l2', n_components=50, verbose=False):
    """
    Computes Kernel Principal Component Analysis (Kernel PCA) on the specified kernel 
    within an AnnData object, updates the object with results, and optionally prints details.

    Parameters:
        data (AnnData): The annotated data matrix to perform Kernel PCA on.
        kernel (str): The kernel to be used for Kernel PCA ('linear', 'poly', 'rbf', 'sigmoid', 'cosine').
        n_components (int): Number of principal components to compute.
        verbose (bool): If True, print computation time and results summary.

    Returns:
        AnnData: The updated AnnData object with Kernel PCA results added.
    """
    if 'time' not in data.uns:
        data.uns['time'] = {}

    # Map 'l2' to 'linear' kernel for computation
    compute_kernel = 'linear' if kernel == 'l2' else kernel

    start_time = time.time()

    # Initialize KernelPCA
    kpca = KernelPCA(n_components=n_components, kernel=compute_kernel, n_jobs=-1)

    # Fit and transform the data
    data_kpca = kpca.fit_transform(data.X)

    # Write results to AnnData object with 'l2' recorded as the kernel
    data.obsm[f'X_kpca_{kernel}'] = data_kpca

    data.uns['time'][f'kpca_{kernel}'] = time.time() - start_time

    if verbose:
        print(f'Kernel PCA ({kernel}): {data_kpca.shape}, running time: {time.time() - start_time}')

    return data

def compute_isomap(data, metric='l2', n_components=50, verbose=False):
    """
    Computes Isomap on the specified distance matrix within
    an AnnData object, updates the object with results, and optionally prints details.

    Parameters:
        data (AnnData): The annotated data matrix to perform Isomap on.
        metric (str): The metric used to create the distance matrix stored in data.obsp.
        n_components (int): Number of components to compute.
        verbose (bool): If True, print computation time and results summary.

    Returns:
        AnnData: The updated AnnData object with Isomap results added.
    """
    if 'time' not in data.uns:
        data.uns['time'] = {}

    start_time = time.time()

    # Initialize Isomap
    isomap = Isomap(n_components=n_components, metric='precomputed')

    # Fit and transform the data
    data_isomap = isomap.fit_transform(data.obsp[f'dm_{metric}'])

    # Write results to AnnData object
    data.obsm[f'X_isomap_{metric}'] = data_isomap
    data.uns['time'][f'isomap_{metric}'] = time.time() - start_time

    if verbose:
        print(f'Isomap ({metric}): {data_isomap.shape}, running time: {time.time() - start_time}')

    return data

def compute_KNN(data, rd_metric = 'pcoa_l2', alg_metric = 'l2'):
    if 'time' not in data.uns:
        data.uns['time'] = {}
    start_time = time.time()

    sc.pp.neighbors(data, n_neighbors = 10, use_rep = f'X_{rd_metric}', metric = alg_metric, key_added = f'KNN_{alg_metric}')
    
    data.uns['time'][f'KNN_{rd_metric}_{alg_metric}'] = time.time() - start_time
    return data

def compute_leiden(data, resolution = 1.0, rd_metric = 'pcoa_l2', alg_metric = 'l2'):
    if 'time' not in data.uns:
        data.uns['time'] = {}
    start_time = time.time()
    
    sc.tl.leiden(data, resolution=resolution, neighbors_key = f'KNN_{alg_metric}', key_added=f'leiden_{rd_metric}_{alg_metric}')
    

    data.uns['time'][f'leiden_{rd_metric}_{alg_metric}'] = time.time() - start_time
    return data

def compute_gmm(data, n_components=10, rd_metric='pcoa_l2', alg_metric='l2'):
    if 'time' not in data.uns:
        data.uns['time'] = {}
    start_time = time.time()

    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(data.X)
    labels = gmm.predict(X)
    data.obs[f'gmm_{rd_metric}_{alg_metric}'] = labels
    
    data.uns['time'][f'gmm_{rd_metric}_{alg_metric}'] = time.time() - start_time
    return data
