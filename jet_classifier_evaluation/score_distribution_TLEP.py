cat score_distribution_TLEP.py
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

plt.figure(figsize=(10, 8))
bins = np.linspace(0, 1, 60)

# Loop over classes using the preloaded arrays
for cname, color, label in zip(flavor_names, colors, plot_labels):
    mask = arrays[f"recojet_is{cname}"] == 1
    scores = score_TLEP[mask]

    plt.hist(scores, bins=bins, histtype='step',
             density=True, linewidth=1.5,
             label=label, color=color)

# Decorations
plt.yscale("log")
plt.xlabel(r"$\mathrm{Score}$", fontsize=14)
plt.ylabel(r"$\mathrm{Probability\ Density}$", fontsize=14)
plt.title(r"$\mathrm{Score\ Distributions\ for\ Leptonic\ Top\ Events}$", fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.ylim(4e-4, 4e2)
plt.xlim(-0.12, 1.12)
plt.margins(x=0.05, y=0.1)
plt.savefig("score_dist_TLEP_all_flavors.png", dpi=300)
plt.show()

