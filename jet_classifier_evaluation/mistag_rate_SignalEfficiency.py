import uproot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

file = uproot.open("pred_Batch2and3.root")
tree = file["Events"]

class_names = ['B', 'C', 'G', 'THAD', 'TLEP', 'WHAD', 'ZHAD', 'UDS']

plot_labels = [
    'b', 'c', 'g', 't_{hadronic}',
    't_{leptonic}', 'W_{hadronic}',
    'Z_{hadronic}', 'uds'
]

colors = ['lightcoral', 'dodgerblue', 'forestgreen', 'deeppink', 
          'mediumorchid', 'goldenrod', 'purple', 'brown']
styles = ['-', '--', '-.', '-', '--', '-', '--', ':']

# preload data for speed
arrays = tree.arrays(
    [f"recojet_is{c}" for c in class_names] +
    [f"score_recojet_is{c}" for c in class_names],
    library="np"
)

plt.figure(figsize=(10, 8))

for i, cname in enumerate(class_names):

    y_true = arrays[f"recojet_is{cname}"]
    y_score = arrays[f"score_recojet_is{cname}"]

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # plot mistag (FPR) vs signal efficiency (TPR)
    plt.plot(tpr, fpr,
             color=colors[i],
             linestyle=styles[i],
             linewidth=2,
             label=f'{plot_labels[i]} (AUC={roc_auc:.3f})')

plt.yscale('log')

plt.xlabel(r'$\mathrm{Signal\ Efficiency}$', fontsize=14)
plt.ylabel(r'$\mathrm{Mistag\ Rate}$', fontsize=14)
plt.title(r'$\mathrm{Mistag\ Rate\ vs.\ Signal\ Efficiency}$', fontsize=16)
plt.xlim([-0.1, 1.1])
plt.ylim([1e-7, 1.6])
plt.grid(True, which='both', alpha=0.3)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("mistag_vs_efficiency.png", dpi=300)
plt.show()
