import uproot
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# Open ROOT file and access tree
file = uproot.open("pred_Batch2and3.root")
tree = file["Events"]

flavor_names = ['B', 'C', 'G', 'THAD', 'TLEP', 'WHAD', 'ZHAD', 'UDS']

plot_labels = [
    'b', 'c', 'g', 't_{hadronic}',
    't_{leptonic}', 'W_{hadronic}',
    'Z_{hadronic}', 'uds'
]

# Define colors and line styles
plot_styles = [
    ('lightcoral', '-'),       # B
    ('dodgerblue', '--'),      # C
    ('forestgreen', '-.'),     # g
    ('deeppink', '-'),         # t_had
    ('mediumorchid', '--'),    # t_lep
    ('goldenrod', '-'),        # W_had
    ('purple', '--'),          # Z_had
    ('gray', ':')              # uds
]

plt.figure(figsize=(12, 10))


# Loop over classes and plot ROC curves
for i, class_name in enumerate(flavor_names):
    # Read numpy arrays directly
    y_true = tree[f'recojet_is{class_name}'].array(library="np")
    y_score = tree[f'score_recojet_is{class_name}'].array(library="np")

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Get color and style
    color, style = plot_styles[i]

    # Plot ROC
    plt.plot(fpr, tpr, color=color, linestyle=style,
             label=f'{plot_labels[i]} (AUC = {roc_auc:.3f})', linewidth=2)

# Plot random baseline
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')

# Axis labels, title, legend
plt.xlim([-0.08, 1.08])
plt.ylim([-0.08, 1.08])
plt.xlabel(r'$\mathrm{False\ Positive\ Rate}$', fontsize=14)
plt.ylabel(r'$\mathrm{True\ Positive\ Rate}$', fontsize=14)
plt.title(r'$\mathrm{ROC\ Curves\ -\ All\ Jet\ Flavors}$', fontsize=16)
plot_labels = [ r'B', r'C', r'G', r't_{\mathrm{hadronic}}', r't_{\mathrm{leptonic}}', r'W_{\mathrm{had}}', r'Z_{\mathrm{had}}', r'UDS']
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)

# Save figure
plt.savefig("roc_all_classes.png", dpi=300, bbox_inches='tight')
plt.show()
