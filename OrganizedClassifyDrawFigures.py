import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import argparse
from itertools import cycle

# Function to read multiple Parquet files and generate combined ROC curves

def evaluate_combined_roc(parquet_files, output_path):
    # Create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.figure(figsize=(10, 6))
    colors = cycle(['#B8D7B3', '#59932D', '#414114', '#51512c'])

    for i, parquet_file in enumerate(parquet_files):
        # Read the Parquet file
        df = pd.read_parquet(parquet_file)

        # Extract necessary data from the dataframe
        roc_data = df['ROC Curve'][0]  # Since ROC data is stored as a dictionary

        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([roc_data['fpr'][key] for key in roc_data['fpr'].keys()]))
        mean_tpr = np.zeros_like(all_fpr)

        for key in roc_data['fpr'].keys():
            fpr = np.array(roc_data['fpr'][key])
            tpr = np.array(roc_data['tpr'][key])
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        mean_tpr /= len(roc_data['fpr'])
        roc_auc = auc(all_fpr, mean_tpr)

        # Plot the macro-average ROC curve for each file
        plt.plot(all_fpr, mean_tpr, lw=2, color=next(colors), label=f'ROC curve {i+1} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=14, fontname='Arial')
    plt.ylabel('True Positive Rate', fontsize=14, fontname='Arial')
    plt.title('Combined Macro-average ROC Curves', fontsize=16, fontname='Arial')
    plt.legend(loc='lower right', fontsize=12, prop={'family': 'Arial'})
    plt.savefig(os.path.join(output_path, 'combined_macro_average_roc_curve.png'))
    plt.close()

    print(f"Combined Macro-average ROC curve saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate combined macro-average ROC curves from multiple test results.')
    parser.add_argument('--parquet_files', type=str, nargs='+', required=True, help='Paths to the Parquet files containing test results.')
    parser.add_argument('--output_path', type=str, required=True, help='Directory to save the combined ROC curve.')
    args = parser.parse_args()

    evaluate_combined_roc(args.parquet_files, args.output_path)