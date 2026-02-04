"""
Visualize continual learning experiment results
"""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set style
sns.set_style('whitegrid')
sns.set_context('talk')

print("Loading results...")
with open("../results/data/continual_learning_results.pkl", "rb") as f:
    results = pickle.load(f)

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ============================================================================
# PLOT 1: Backward Transfer Comparison
# ============================================================================

ax1 = axes[0]

metrics = ['AUC_ROC', 'Average_Precision']
bt_correlational = [results['backward_transfer'][m]['correlational'] for m in metrics]
bt_causal = [results['backward_transfer'][m]['causal'] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, bt_correlational, width, label='Correlational', color='#e74c3c', alpha=0.8)
bars2 = ax1.bar(x + width/2, bt_causal, width, label='Causal', color='#2ecc71', alpha=0.8)

ax1.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax1.set_ylabel('Backward Transfer', fontsize=14)
ax1.set_title('Forgetting After Concept Drift\n(Higher is Better)', fontsize=16, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(['AUC ROC', 'Average Precision'], fontsize=12)
ax1.legend(fontsize=12)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.3f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')

# ============================================================================
# PLOT 2: Performance Over Time
# ============================================================================

ax2 = axes[1]

# Create timeline data
timeline_data = {
    'Stage': ['Period 1\n(before CL)', 'Period 2\n(Forward Transfer)', 'Period 1\n(after CL)'],
    'Correlational': [
        results['correlational']['period1_before_cl']['AUC_ROC'],
        results['correlational']['period2_after_cl']['AUC_ROC'],
        results['correlational']['period1_after_cl']['AUC_ROC']
    ],
    'Causal': [
        results['causal']['period1_before_cl']['AUC_ROC'],
        results['causal']['period2_after_cl']['AUC_ROC'],
        results['causal']['period1_after_cl']['AUC_ROC']
    ]
}

x_positions = [0, 1, 2]

ax2.plot(x_positions, timeline_data['Correlational'], 'o-',
         color='#e74c3c', linewidth=3, markersize=10, label='Correlational', alpha=0.8)
ax2.plot(x_positions, timeline_data['Causal'], 's-',
         color='#2ecc71', linewidth=3, markersize=10, label='Causal', alpha=0.8)

# Add shaded region for concept drift
ax2.axvspan(0.8, 1.2, alpha=0.1, color='orange', label='Concept Drift')

ax2.set_xticks(x_positions)
ax2.set_xticklabels(timeline_data['Stage'], fontsize=11)
ax2.set_ylabel('AUC ROC', fontsize=14)
ax2.set_title('Performance Over Time\n(Concept Drift at Period 2)', fontsize=16, fontweight='bold')
ax2.set_ylim([0.8, 1.05])
ax2.legend(fontsize=12, loc='lower left')
ax2.grid(True, alpha=0.3)

# Add annotations
ax2.annotate('Catastrophic\nForgetting!',
            xy=(2, timeline_data['Correlational'][2]),
            xytext=(2.3, 0.87),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2),
            fontsize=11, color='#e74c3c', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e74c3c', alpha=0.8))

ax2.annotate('No Forgetting!',
            xy=(2, timeline_data['Causal'][2]),
            xytext=(2.3, 1.02),
            arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2),
            fontsize=11, color='#2ecc71', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2ecc71', alpha=0.8))

plt.tight_layout()

# Save figure
output_file = "../results/figures/continual_learning_results.png"
import os
os.makedirs("../results/figures", exist_ok=True)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_file}")

# ============================================================================
# PLOT 3: Summary Statistics Table
# ============================================================================

fig2, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Create table data
table_data = []
table_data.append(['', 'Correlational BT', 'Causal BT', 'Improvement'])
table_data.append(['AUC ROC',
                  f"{results['backward_transfer']['AUC_ROC']['correlational']:+.4f}",
                  f"{results['backward_transfer']['AUC_ROC']['causal']:+.4f}",
                  f"+{results['backward_transfer']['AUC_ROC']['difference']:.4f}"])
table_data.append(['Average Precision',
                  f"{results['backward_transfer']['Average_Precision']['correlational']:+.4f}",
                  f"{results['backward_transfer']['Average_Precision']['causal']:+.4f}",
                  f"+{results['backward_transfer']['Average_Precision']['difference']:.4f}"])
table_data.append(['Card Precision@100',
                  f"{results['backward_transfer']['Card_Precision@100']['correlational']:+.4f}",
                  f"{results['backward_transfer']['Card_Precision@100']['causal']:+.4f}",
                  f"+{results['backward_transfer']['Card_Precision@100']['difference']:.4f}"])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.3, 0.25, 0.25, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.5)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code results
for i in range(1, 4):
    # Correlational (red-ish)
    table[(i, 1)].set_facecolor('#ffcccc')
    # Causal (green-ish)
    table[(i, 2)].set_facecolor('#ccffcc')
    # Improvement (blue-ish)
    table[(i, 3)].set_facecolor('#cce5ff')

ax.set_title('Backward Transfer Summary\n(Change in Performance After Continual Learning)',
            fontsize=16, fontweight='bold', pad=20)

table_file = "../results/figures/backward_transfer_table.png"
plt.savefig(table_file, dpi=300, bbox_inches='tight')
print(f"Table saved to: {table_file}")

plt.show()

print("\n[SUCCESS] Visualizations created!")
print("\nKey Finding:")
print(f"  Causal model improvement: +{results['backward_transfer']['Average_Precision']['difference']:.4f} on Average Precision")
print(f"  This means {abs(results['backward_transfer']['Average_Precision']['difference'])*100:.1f}% less forgetting!")
