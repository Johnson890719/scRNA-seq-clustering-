# Robust Evaluation of Clustering Algorithms in scRNA-seq

This repository implements a modular framework for evaluating clustering algorithms on single-cell RNA-seq data using a variety of **similarity and distance metrics**. Starting from preprocessed `.h5ad` files, the pipeline supports custom sampling, dimensionality reduction, and benchmarking across multiple clustering algorithms.

A key focus of this work is a systematic comparison between **distance-based** (e.g., Euclidean, Manhattan) and **similarity-based** (e.g., Cosine similarity, Pearson correlation) metrics and their robustness under noise and varying ground truth label quality.

---
## Schematic Diagram
![image](https://github.com/user-attachments/assets/602f5768-6395-49a5-8328-a8b899e1ae9d)

## Project Structure

```bash
scRNA-seq-clustering/
├── Experiments/               # Main Jupyter notebooks
│   ├── 0.data_loading.ipynb         # Load h5ad, normalize, log-transform, HVG selection
│   ├── 1.SamplingFramework.ipynb    # Downsampling strategy
│   ├── 2.PCA_vs_PCOA-TMS.ipynb      # Compare DR methods
│   └── 3.clustering_pipeline.ipynb  # Run clustering algorithms & similarity metrics
│
├── demo/                     # Baseline pipeline using Scanpy
│
├── packages/                 # Modular Python scripts
│   ├── data_loader.py, clustering.py, utils.py, ...
│   └── setup.py              # Optional: pip install -e packages/
│
├── result/                   # Analysis notebooks and figures
│   ├── Gold vs. Silver.ipynb     # Impact of label quality
│   ├── Noise.ipynb               # Noise robustness tests
│   ├── Overall_plot.ipynb
│   └── Paper_Figure.ipynb        # Final plots
│
├── sbatch/                   # SLURM job scripts for HPC
└── README.md
```

##  Key Features

- Systematic benchmarking of clustering performance across different **similarity/distance metrics**  
- Supports multiple clustering algorithms (KMeans, Leiden, Spectral, etc.)  
- Dimensionality reduction comparison  
- Ground truth label evaluation: Gold vs. Silver standard  
- Noise injection experiments to test metric robustness  
- HPC-compatible job submission via `.sbatch` scripts  


## Key Experimental Findings
1. Without Dimensionality Reduction: Correlation Metrics Shine
- Correlation-based metrics (e.g., Cosine, Pearson) outperform distance-based metrics like L1 and L2 in raw feature space.
- Across algorithms like Spectral, KMeans, and BanditPAM, L1 tends to perform significantly worse than L2.
- Paired t-tests show a statistically significant ΔNMI improvement using Pearson over L2.
![image](https://github.com/user-attachments/assets/2d80ac10-50d4-483f-804e-71e6e9c6304f)

2. After PCA: Differences Flatten
- After PCA, the performance gap between correlation and distance metrics decreases.
- Significant ΔNMI gains using Pearson over L1 are observed mainly in Hierarchical and Leiden clustering.
- PCA appears to mitigate the influence of distance/similarity metric choices.
![image](https://github.com/user-attachments/assets/d47b7906-fad1-42d7-89cf-0ff348f0ce4e)

3. Across Dimensionality Reduction Methods: Correlation Metrics Still Advantageous
- DR methods like PCoA, Isomap, and Kernel PCA show continued (but varied) benefit from correlation-based metrics.
- Hierarchical clustering consistently benefits the most from correlation metrics.
- The effect is clearest in PCoA and Isomap, with significant ΔNMI gains across multiple clustering algorithms.
![image](https://github.com/user-attachments/assets/96a97b5e-fbbb-4920-a0c5-09561e90ce62)

4. Combined Use of Similarity Metrics in DR + Clustering May Backfire
- Regression analysis reveals negative effect sizes when similarity metrics are applied simultaneously to both DR and clustering.
- Using similarity-based metrics only in one step (DR or clustering) yields neutral to slightly positive improvements.
- Interpretation: Applying similarity metrics in DR may already extract key relational patterns, making repeated use redundant or even detrimental.
![image](https://github.com/user-attachments/assets/4655a468-5feb-478d-8749-70e6a4de806c)
