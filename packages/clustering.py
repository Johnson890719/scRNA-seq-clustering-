import time
import scanpy as sc
from banditpam import KMedoids


def pca(adata, svd_solver='arpack'):
    
    """ perform PCA on scRNA,

    Args: 
        adata: anntotaed of scRNA

     

    """
    sc.tl.pca(adata, svd_solver='arpack')

def neighbors(adata, n_neighbors = 10, use_pca = True):
    if use_pca:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=40)
    else:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=0)

def cluster_leiden(adata):
    start_time = time.time()
    sc.tl.leiden(adata)
    print('leiden completed, time=%0.1fs' % (time.time() - start_time))

def cluster_banditpam(adata, n_medoids = 5, metric = 'L1'):
    start_time = time.time()
    kmed = KMedoids(n_medoids=n_medoids, algorithm="BanditPAM")
    kmed.fit(adata.X, metric)
    adata.obs['kmed_'+metric] = kmed.labels
    adata.obs['kmed_'+metric] = adata.obs['kmed_'+metric].astype('category')
    print('kmed_%s (K = %d) completed, time=%0.1fs' % (metric, n_medoids, (time.time() - start_time)))

def cluster_banditpam_l1(adata, n_medoids = 5, metric = 'L1'):
    start_time = time.time()
    kmed = KMedoids(n_medoids=n_medoids, algorithm="BanditPAM")
    kmed.fit(adata.X, metric)
    adata.obs['kmed_l1'] = kmed.labels
    adata.obs['kmed_l1'] = adata.obs['kmed_l1'].astype('category')
    print('kmed_l1 completed, time=%0.1fs' % (time.time() - start_time))

def cluster_banditpam_l2(adata, n_medoids = 5, metric = 'L2'):
    start_time = time.time()
    kmed = KMedoids(n_medoids=n_medoids, algorithm="BanditPAM")
    kmed.fit(adata.X, 'L2')
    adata.obs['kmed_l2'] = kmed.labels
    adata.obs['kmed_l2'] = adata.obs['kmed_l2'].astype('category')
    print('kmed_l2 completed, time=%0.1fs' % (time.time() - start_time))
