import os
import matplotlib.pyplot as plt
import numpy as np

def plot_cross_validation_results(cv_summary, output_path="./plots/cross_validation_results.png"):
    """
    Visualize cross-validation results for each metric using a scatter plot with error bars.

    Args:
        cv_summary (dict): A dictionary where keys are metric names, and values are dictionaries
                           with 'mean' and 'std' keys.
        output_path (str): The file path to save the plot.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare data for the scatter plot
    metrics = list(cv_summary.keys())  # e.g., ['accuracy', 'loss']
    means = [cv_summary[metric]['mean'] for metric in metrics]
    stds = [cv_summary[metric]['std'] for metric in metrics]

    # Create the scatter plot
    x_positions = np.arange(len(metrics))  # x-axis positions for each metric
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        x_positions, means, yerr=stds, fmt='o', color='blue', ecolor='red', capsize=5, label='Mean ± Std'
    )

    # Add labels and formatting
    plt.xticks(x_positions, metrics, rotation=0, fontsize=10)
    plt.title('Cross-Validation Results with Error Bars')
    plt.xlabel('Metrics')
    plt.ylabel('Mean Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    # Save and close the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
