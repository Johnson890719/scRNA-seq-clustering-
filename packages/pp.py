import scanpy as sc

def preprocess(adata, min_genes = 500, min_cells = 2000,  max_counts = None, target_sum = 1e4, verbose = True):
    
    adata.var_names_make_unique()
    
    sc.pp.filter_cells(adata, min_genes = min_genes)
    if max_counts:
        sc.pp.filter_cells(adata, max_counts = max_counts)
    sc.pp.filter_genes(adata, min_cells = min_cells)

    # annotate the group of mitochondrial genes as 'mt'
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # normalization
    sc.pp.normalize_total(adata, target_sum=target_sum)

    # log-transform
    sc.pp.log1p(adata)

    if verbose:
        print('n_obs × n_vars: ', adata.shape)
        
    return adata

def hvg(adata, verbose = True):
    sc.pp.highly_variable_genes(adata,
                                n_top_genes=2000,
                                subset=True)
    if verbose:
        print('n_obs × n_vars: ', adata.shape)
    return adata
