import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib import cm
from sklearn.metrics import r2_score

def annotate_metrics(x, y, ax=None, **kwargs):
    """
    Annotate MAE, RMSE, and R² directly on the graph as red text.
    """
    ax = ax or plt.gca()
    mae = np.mean(np.abs(x - y))
    rmse = np.sqrt(np.mean((x - y) ** 2))
    r2 = r2_score(x, y)  # Using sklearn

    ax.text(0.05, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}',
            transform=ax.transAxes, fontsize=6, color='red', ha='left', va='top')

# Load the DataFrame
df = pd.read_parquet('/mnt/storage/nobackup/nca121/paper_charge_comparisons/async_chargecraft_more_workers/charge_models_test.parquet')

def scatter_plot(x, y, **kwargs):
    return plt.scatter(x, y, **kwargs)

# Define charge models and flatten the arrays
charge_columns = ['am1bcc_charges', 'riniker_monopoles', 'espaloma_charges', 'resp_charges']
mbis_charges_flat = np.concatenate(df['mbis_charges'].values)

# Create a DataFrame for pairwise comparison with MBIS charges
charges_df = pd.DataFrame({'MBIS Charges': mbis_charges_flat})

# Add flattened data for each charge model
for col in charge_columns:
    new_col_name = col.replace('_charges', '').replace('_monopoles', '').capitalize()
    charges_df[new_col_name] = np.concatenate(df[col].values)

print(charges_df.columns)
# Rename columns according to your desired labels
charges_df = charges_df.rename(columns={
    "MBIS Charges": "MBIS",
    "Am1bcc": "AM1-BCC",
    "Riniker": "Multipole GNN",
    "Resp": "RESP",
    "espaloma":"espaloma-charge"
})

desired_order = ["AM1BCC", "RESP", "espaloma-charge", "Multipole GNN"]
charges_df = charges_df[desired_order]
# Initialize the PairGrid
g = sns.PairGrid(charges_df, height=3, aspect=1.2, diag_sharey=False, corner=True)

# Plot scatter points
g.map_offdiag(scatter_plot, s=0.5, alpha=0.7, color='blue')

# Add equality line y=x
for ax_row in g.axes:
    for ax in ax_row:
        if ax is not None:
            ax.plot([-1.5, 1.5], [-1.5, 1.5], color='gray', linestyle='--', linewidth=0.8)

# Annotate metrics on off-diagonal
g.map_offdiag(annotate_metrics)

# Remove the diagonal axes
for ax in np.diag(g.axes):
    ax.set_visible(False)

# Adjust axes
g.set(xlim=(-1.5, 2), ylim=(-1.5, 2))
# for ax in g.axes.flat:
#     if ax is not None:
#         ax.set_xlabel('')
#         ax.set_ylabel('')

g.fig.subplots_adjust(top=0.9)
# g.fig.suptitle('Pairwise Comparison of Charge Models', fontsize=16)

plt.savefig('pairwise_partial_charges_density_2.png', dpi=300, bbox_inches='tight')
plt.show()
