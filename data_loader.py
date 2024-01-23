import anndata
import numpy as np


def load_annData(file_path, format = "h5ad"):
    
    """load processed data
    
    Args:
        file_path (str): scRNA count matrix data path
        format (str): format the data used to store with
        
    Returns:
        adata (AnnData): Combined data for FACS and droplet
    """

    return anndata.read_h5ad(file_path)
    
    