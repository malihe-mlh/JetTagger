import uproot
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Load ROOT file
file = uproot.open("pred_Batch2and3.root")
tree = file["Events"]

class_names = ['B', 'C', 'G', 'THAD', 'TLEP', 'WHAD', 'ZHAD', 'UDS']

# --- Load true labels ---
y_true_matrix = np.vstack([
    tree[f"recojet_is{c}"].array(library="np")
    for c in class_names
]).T

y_true = np.argmax(y_true_matrix, axis=1)

# --- Load predicted scores ---
y_score_matrix = np.vstack([
    tree[f"score_recojet_is{c}"].array(library="np")
    for c in class_names
]).T

y_pred = np.argmax(y_score_matrix, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, normalize='true')

plot_labels = [
    r'$b$', r'$c$', r'$g$', r'$t_{\mathrm{hadronic}}$',
    r'$t_{\mathrm{leptonic}}$', r'$W_{\mathrm{hadronic}}$',
    r'$Z_{\mathrm{hadronic}}$', r'$uds$'
]

plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap='Blues')
plt.colorbar(label=r'$\mathrm{Fraction}$')

# Tick labels with rotation
plt.xticks(np.arange(len(class_names)), labels=plot_labels, rotation=45, ha='right', fontsize=12)
plt.yticks(np.arange(len(class_names)), labels=plot_labels, fontsize=12)

plt.xlabel(r'$\mathrm{Predicted\ Flavor}$', fontsize=14)
plt.ylabel(r'$\mathrm{True\ Flavor}$', fontsize=14)
plt.title(r'$\mathrm{Multi\ Flavor\ Confusion\ Matrix}$', fontsize=16)

# Add values
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm[i, j]:.3f}", ha="center", va="center", fontsize=10)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()
