import argparse
import anndata
from pipeline import cl

def main(dataset_file, output_path):
    data = anndata.read_h5ad(dataset_file)
    print(dataset_file)

    # Compute Isomap for different metrics
    for metric in ['l1', 'l2', 'cosine', 'correlation']:
        print(f'[{metric}]')
        data = cl.compute_distance_matrix(data, metric, verbose=True)  # N x N
        data = cl.compute_isomap(data, metric, 50, verbose=True)  # N x K

    data.write(dataset_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Isomap on a dataset.")
    parser.add_argument('--dataset_file', type=str, required=True, help="Path to the input dataset file.")
    parser.add_argument('--output_path', type=str, required=True, help="Directory to save the output files.")
    
    args = parser.parse_args()
    main(args.dataset_file, args.output_path)