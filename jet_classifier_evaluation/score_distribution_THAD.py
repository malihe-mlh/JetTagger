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
    ["score_recojet_isTHAD"] + [f"recojet_is{f}" for f in flavor_names],
    library="np"
)

score_THAD = arrays["score_recojet_isTHAD"]

plt.figure(figsize=(10, 8))
bins = np.linspace(0, 1, 60)

# Loop over classes using the preloaded arrays
for cname, color, label in zip(flavor_names, colors, plot_labels):
    mask = arrays[f"recojet_is{cname}"] == 1
    scores = score_THAD[mask]

    plt.hist(scores, bins=bins, histtype='step',
             density=True, linewidth=1.5,
             label=label, color=color)

# Decorations
plt.yscale("log")
plt.xlabel(r"$\mathrm{Score}$", fontsize=14)
plt.ylabel(r"$\mathrm{Probability\ Density}$", fontsize=14)
plt.title(r"$\mathrm{Score\ Distributions\ for\ Hadronic\ Top\ Events}$", fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(4e-4, 4e2)
plt.xlim(-0.12, 1.12)
plt.margins(x=0.05, y=0.1)
plt.tight_layout()
plt.savefig("score_dist_THAD_all_flavors.png", dpi=300)
plt.show()
