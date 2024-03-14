import matplotlib.pyplot as plt
import numpy as np

# Data
models = ["ResNet152", "Wide ResNet101", "VGG", "ResNet50", "ResNet18", "EfficientNet", "RegNet", "ConvNext"]
fpr_means_white = [0.8765, 0.54425, 0.8055, 0.5255, 0.7615, 0.73907, 0.8711, 0.5068]
fpr_stds_white = [0.08847, 0.10654, 0.06440, 0.09277, 0.08774, 0.09020, 0.08313, 0.12391]
tpr_means_white = [0.8614, 0.62345, 0.9010, 0.6772, 0.7652, 0.81995, 0.82334, 0.67951]
tpr_stds_white = [0.03166, 0.04687, 0.02581, 0.04091, 0.03862, 0.03885, 0.03098, 0.03624]

fpr_means_non_white = [0.8032, 0.54165, 0.7496, 0.3792, 0.6493, 0.7519, 0.8129, 0.4247]
fpr_stds_non_white = [0.13768, 0.12543, 0.12078, 0.11891, 0.10568, 0.11883, 0.08817, 0.12211]
tpr_means_non_white = [0.8339, 0.6161, 0.8607, 0.6464, 0.7396, 0.8053, 0.7901, 0.6696]
tpr_stds_non_white = [0.05527, 0.09524, 0.05875, 0.09321, 0.08348, 0.06672, 0.08455, 0.10020]

# Number of models
n_models = len(models)

# X locations for the groups
ind = np.arange(n_models)

print(ind * 2)

# Bar width
width = 0.35

# Plotting
fig, ax = plt.subplots(figsize=(15, 8))

# FPR bars
fpr_bars_white = ax.bar(2 * ind - width/2, fpr_means_white, width, yerr=fpr_stds_white, label='FPR White', capsize=5, alpha=0.6)
fpr_bars_non_white = ax.bar(2 * ind + width/2, fpr_means_non_white, width, yerr=fpr_stds_non_white, label='FPR Non-White', capsize=5, alpha=0.6)

# TPR bars

tpr_bars_white = ax.bar(2 * ind + width / 2 + width, tpr_means_white, width, yerr=fpr_stds_white, label='TPR White', capsize=5, alpha=0.6)
tpr_bars_non_white = ax.bar(2 * ind + width/2 + 2*width, tpr_means_non_white, width, yerr=fpr_stds_non_white, label='TPR Non-White', capsize=5, alpha=0.6)

# ax.bar(ind - width/2, tpr_means_white, yerr=tpr_stds_white, fmt='o', color='r', label='TPR White')
# ax.bar(ind + width/2, tpr_means_non_white, yerr=tpr_stds_non_white, fmt='o', color='g', label='TPR Non-White')

# Labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Rates')
ax.set_title('FPR and TPR by Model and Group')
ax.set_xticks(ind * 2)
ax.set_xticklabels(models)
ax.legend()

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()