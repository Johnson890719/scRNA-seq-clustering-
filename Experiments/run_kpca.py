import argparse
import anndata
from pipeline import cl

def main(dataset_file, output_path):
    data = anndata.read_h5ad(dataset_file)
    print(dataset_file)

    for kernel in ['l2', 'cosine']:
        print(f'[Kernel PCA - {kernel}]')
        data = cl.compute_kernelPCA(data, kernel, 50, verbose=True)  # N x K

    data.write(dataset_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Isomap and Kernel PCA on a dataset.")
    parser.add_argument('--dataset_file', type=str, required=True, help="Path to the input dataset file.")
    parser.add_argument('--output_path', type=str, required=True, help="Directory to save the output files.")
    
    args = parser.parse_args()
    main(args.dataset_file, args.output_path)
