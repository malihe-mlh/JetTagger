import uproot
import numpy as np
import matplotlib.pyplot as plt

# Load ROOT file
file = uproot.open("pred_Batch2and3.root")
tree = file["Events"]

# Class names and colors
flavor_names = ['B', 'C', 'G', 'THAD', 'TLEP', 'WHAD', 'ZHAD', 'UDS']
colors = ['lightcoral', 'dodgerblue', 'forestgreen', 'deeppink', 
          'mediumorchid', 'goldenrod', 'purple', 'brown']
plot_labels = [
    'b', 'c', 'g', 't_{hadronic}',
    't_{leptonic}', 'W_{hadronic}',
    'Z_{hadronic}', 'uds'
]

# Load all necessary arrays at once
arrays = tree.arrays(
    ["score_recojet_isTLEP"] + [f"recojet_is{f}" for f in flavor_names],
    library="np"
)

score_TLEP = arrays["score_recojet_isTLEP"]

# Prepare data for stacked histogram
scores_list = []
labels_list = []
colors_list = []

for cname, color, label in zip(flavor_names, colors, plot_labels):
    mask = arrays[f"recojet_is{cname}"] == 1
    scores = score_TLEP[mask]
    scores_list.append(scores)
    labels_list.append(label)
    colors_list.append(color)

# Plot stacked histogram,  density=True
plt.figure(figsize=(12, 8))
bins = np.linspace(0, 1, 60)

plt.hist(scores_list, bins=bins, stacked=True,
         color=colors_list, label=labels_list, alpha=0.8,
         density=True)

plt.xlabel(r'$\mathrm{Score}$', fontsize=14)
plt.ylabel(r'$\mathrm{Probability\ Density}$', fontsize=14)
plt.title(r"$\mathrm{Score\ Distributions\ for\ Leptonic\ Top\ Events}$", fontsize=16)

plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.ylim(4e-4, 4e2)
plt.xlim(-0.12, 1.12)

plt.tight_layout()
plt.savefig("TLEP_score_stacked_density.png", dpi=300)
plt.show()
