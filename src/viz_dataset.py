import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# Read the CSV file with items for CHR paper
dataset = pd.read_csv('./paper/data/item_selection.csv')

# Define era order
era_order = ['Frühgeschichte', 'Antike', 'Mittelalter', 'Frühe Neuzeit', 
             '19. Jahrhundert', '20. Jahrhundert', '21. Jahrhundert']

# Create a cross-tabulation of Type and Era with normalized values
cross_tab = pd.crosstab(dataset['Type'], dataset['Era'], normalize='index') * 100

# Reorder columns according to era_order
cross_tab = cross_tab.reindex(columns=era_order)

# Specify SGB colours for eras
era_colors = {
     'Frühgeschichte': '#cb7501',
     'Antike': '#ad7800',
     'Mittelalter': '#008487',
     'Frühe Neuzeit': '#e51e3c',
     '19. Jahrhundert': '#bf509b',
     '20. Jahrhundert': '#0080c7',
     '21. Jahrhundert': '#d3480a'
 }

# add grey as fallback in case of column mismatch
colors_list = [era_colors.get(col, '#cccccc') for col in cross_tab.columns]

# Set up stacked bar chart
fig1, ax = plt.subplots(figsize=(12, 8))
cross_tab.plot(kind='barh', stacked=True, color=colors_list, ax=ax, legend=False)

# Customize the plot
ax.set_title('Distribution of Item Types Across Eras (full selection, n=100)', pad=20)
ax.set_ylabel('')
ax.set_xlabel('')
ax.tick_params(axis='y', rotation=0)

# x-axis formatting
ax.set_xlim(0, 100)
ax.xaxis.set_major_locator(plt.MultipleLocator(20))
ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}%"))
ax.set_axisbelow(True)
ax.grid(axis='x', color='lightgrey', linestyle='--', linewidth=0.8)

# Add a figure-level legend below the plot
handles, labels = ax.get_legend_handles_labels()
fig1.legend(handles, labels, title='Era',
            loc='lower center',
            bbox_to_anchor=(0.0, -0.12, 1.0, 0.08),
            ncol=len(era_order), frameon=False, mode='expand', bbox_transform=fig1.transFigure, borderaxespad=0.5)

# Adjust layout and save the first figure
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig1.savefig('./paper/images/fig_type_era_full.png', dpi=300, bbox_inches='tight')

# --- Plot with Survey-Subset only ---
# Build survey subset
if 'survey' in dataset.columns:
    subset = dataset[dataset['survey'] == 1]
else:
    subset = dataset.iloc[0:0]

# Compute percent-normalized crosstab for the survey subset and align with full table
cross_tab_survey = pd.crosstab(subset['Type'], subset['Era'], normalize='index') * 100
cross_tab_survey = cross_tab_survey.reindex(index=cross_tab.index, columns=era_order).fillna(0)

# Create second figure with and axis and plot using the same colors and formatting
fig2, ax2 = plt.subplots(figsize=(12, 8))
cross_tab_survey.plot(kind='barh', stacked=True, color=colors_list, ax=ax2, legend=False)

# Title and axis labels
ax2.set_title(f'Distribution of Item Types Across Eras (survey subset, n={len(subset)})', pad=20)
ax2.set_ylabel('')
ax2.set_yticklabels(cross_tab.index)
ax2.tick_params(axis='y', rotation=0)

# x-axis formatting
ax2.set_xlim(0, 100)
ax2.xaxis.set_major_locator(plt.MultipleLocator(20))
ax2.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}%"))
ax2.set_axisbelow(True)
ax2.grid(axis='x', color='lightgrey', linestyle='--', linewidth=0.8)

# Add a figure-level legend below the plot
fig2 = ax2.get_figure()
handles2, labels2 = ax2.get_legend_handles_labels()
fig2.legend(handles2, labels2, title='Era',
            loc='lower center',
            bbox_to_anchor=(0.0, -0.12, 1.0, 0.08),
            ncol=len(era_order), frameon=False, mode='expand', bbox_transform=fig2.transFigure, borderaxespad=0.5)

# Adjust layout and save survey-only figure
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig2.savefig('./paper/images/fig_type_era_subset.png', dpi=300, bbox_inches='tight')
