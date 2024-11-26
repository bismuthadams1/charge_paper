import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib import cm

# Load the DataFrame
df = pd.read_parquet('/mnt/storage/nobackup/nca121/paper_charge_comparisons/async_chargecraft_more_workers/charge_models_test.parquet')

# Define the density scatter plot function
def density_scatter_plot(x, y, **kwargs):
    """
    Plot a scatter plot with point colors based on density.
    """
    # Kernel Density Estimate (KDE)
    values = np.vstack((x, y))
    kernel = gaussian_kde(values)
    kde = kernel.evaluate(values)

    # Create array with colors for each data point
    norm = Normalize(vmin=kde.min(), vmax=kde.max())
    colors = cm.ScalarMappable(norm=norm, cmap='viridis').to_rgba(kde)

    # Override original color argument
    kwargs['color'] = colors
    return plt.scatter(x, y, **kwargs)

# Define MAE and RMSE annotation function
def annotate_metrics(x, y, ax=None, **kwargs):
    """
    Annotate MAE and RMSE directly on the graph as plain text.
    """
    mae = np.mean(np.abs(x - y))
    rmse = np.sqrt(np.mean((x - y) ** 2))
    ax = ax or plt.gca()
    ax.text(0.05, 0.9, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}',
            transform=ax.transAxes, fontsize=6, color='darkblue', ha='left', va='top')

# Define charge models and flatten the arrays
charge_columns = ['am1bcc_charges', 'riniker_monopoles', 'espaloma_charges', 'resp_charges']
mbis_charges_flat = np.concatenate(df['mbis_charges'].values)

# Create a DataFrame for pairwise comparison with MBIS charges
charges_df = pd.DataFrame({'MBIS Charges': mbis_charges_flat})

# Add flattened data for each charge model
for col in charge_columns:
    charges_df[col.replace('_charges', '').replace('_monopoles', '').capitalize()] = np.concatenate(df[col].values)

# Initialize the PairGrid for pairwise comparison
g = sns.PairGrid(charges_df, height=3, aspect=1.2)

# Use density scatter plot for off-diagonal comparisons
g.map_offdiag(density_scatter_plot, s=2, alpha=0.7)  # Smaller points, transparent

# Add the equality line (y = x) to all off-diagonal plots
for ax in np.ravel(g.axes):
    if ax is not None:  # Check if the axis exists
        ax.plot([-1.5, 1.5], [-1.5, 1.5], color='gray', linestyle='--', linewidth=0.8)

# Annotate MAE and RMSE
g.map_offdiag(annotate_metrics)

# Remove diagonal plots
for ax in np.diag(g.axes):
    ax.set_visible(False)

# Adjust labels and layout
g.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
g.set_titles("{col_name}")
g.set_xlabels('')
g.set_ylabels('')
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Pairwise Comparison of Charge Models', fontsize=16)

# Save and display the plot
plt.savefig('pairwise_partial_charges_density.png', dpi=300, bbox_inches='tight')
plt.show()
