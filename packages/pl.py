import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

def plot_scree(eigvals):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(eigvals) + 1), eigvals, 'o-', color='b')
    ax.set_title('Scree Plot of PCoA Eigenvalues')
    ax.set_xlabel('Principal Coordinate Index')
    ax.set_ylabel('Eigenvalue')

    # Calculate the cumulative variance
    cumulative_variance = np.cumsum(eigvals) / np.sum(eigvals)
    ax2 = ax.twinx()
    ax2.plot(range(1, len(eigvals) + 1), cumulative_variance, 'o-', color='r')
    ax2.set_ylabel('Cumulative Proportion of Variance', color='r')
    
    # Draw horizontal lines for the 90% and 99% variance explained thresholds
    ax2.axhline(y=0.90, color='g', linestyle='--', label='90%')
    ax2.axhline(y=0.99, color='purple', linestyle='--', label='99%')

    # Adding legend to the right axis
    ax2.legend(loc='lower right')

    # Color red axis labels
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    plt.show()

def plot_cell_types(adata, target):
    cell_types = Counter(adata.obs[target])
    
    df = pd.DataFrame(list(cell_types.items()), columns=['Cell Type', 'Count'])
    
    # Sort by count for better visualization
    df = df.sort_values(by='Count', ascending=False)
    
    # Set the figure size and style
    plt.figure(figsize=(20, 10))
    sns.set(style="whitegrid")
    
    # Create a bar plot
    ax = sns.barplot(x='Count', y='Cell Type', data=df, palette='viridis')
    
    # Set labels and title
    ax.set_xlabel('Count')
    ax.set_ylabel('Cell Type')
    ax.set_title('Counts of Different Cell Types - TMS_fasq')
    
    # Create new y-tick labels with ranking
    y_labels = [f'{i+1}. {cell_type[:30]}' for i, cell_type in enumerate(df['Cell Type'])]
    ax.set_yticklabels(y_labels, horizontalalignment='left')
    ax.tick_params(axis='y', which='major', pad=200)
    ax.set_xticks([200, 400, 500])
    plt.show()