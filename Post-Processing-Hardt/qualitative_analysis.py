import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Provided data
models = ['ResNet152', 'Wide_ResNet101', 'VGG', 'ResNet50', 'ResNet18', 'EfficientNet', 'RegNet', 'ConvNext']
groups = ['White', 'Non-White']
metrics = ['FPR', 'TPR', 'Loss']

# Data from the table
pre_adjust_fpr = [0.8889, 0.6786, 0.4441, 0.1071, 0.8000, 0.8214, 0.5333, 0.5537, 0.7556, 0.8571, 0.7556, 0.7143,
                  0.8667, 1.0000, 0.5111, 0.6786]
post_adjust_fpr = [0.8889, 0.7500, 0.5556, 0.0745, 0.8000, 0.6786, 0.5333, 0.4286, 0.7556, 0.7500, 0.7556, 0.7500,
                   0.8667, 0.8214, 0.5111, 0.3571]

pre_adjust_tpr = [0.8667, 0.8214, 0.5556, 0.2636, 0.8000, 0.8899, 0.5333, 0.6930, 0.7556, 0.8148, 0.7556, 0.7865,
                  0.9778, 0.9630, 1.0000, 0.8148]
post_adjust_tpr = [0.8652, 0.8148, 0.6267, 0.2955, 0.8889, 0.8599, 0.6742, 0.5926, 0.7790, 0.7037, 0.8165, 0.8148,
                   0.9704, 0.8148, 0.6704, 0.7407]

pre_adjust_loss = [0.3616, 0.3161, 0.4033, 0.4042, 0.2371, 0.2289, 0.3706, 0.3651, 0.3188, 0.3324, 0.3188, 0.2970,
                   0.1989, 0.3106, 0.3678, 0.3488]
post_adjust_loss = [0.2752, 0.2752, 0.4142, 0.4142, 0.2289, 0.2289, 0.3651, 0.3651, 0.3324, 0.3324, 0.2970, 0.2970,
                    0.3106, 0.3106, 0.3488, 0.3488]

# Split the metrics by group
metrics_by_group = {
    group: {
        'FPR': {'Pre-Adjust': pre_adjust_fpr[i::2], 'Post-Adjust': post_adjust_fpr[i::2]},
        'TPR': {'Pre-Adjust': pre_adjust_tpr[i::2], 'Post-Adjust': post_adjust_tpr[i::2]},
        'Loss': {'Pre-Adjust': pre_adjust_loss[i::2], 'Post-Adjust': post_adjust_loss[i::2]},
    }
    for i, group in enumerate(groups)
}

# Plotting the data
fig, axs = plt.subplots(3, 1, figsize=(14, 18))

for i, metric in enumerate(metrics):
    ax = axs[i]
    ax.set_title(f'{metric} for Different Models')
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models)
    ax.set_xlabel('Models')
    ax.set_ylabel(metric)

    for group, group_data in metrics_by_group.items():
        ax.plot(models, group_data[metric]['Pre-Adjust'], marker='o', linestyle='-', label=f'{group} Pre-Adjust')
        ax.plot(models, group_data[metric]['Post-Adjust'], marker='x', linestyle='--', label=f'{group} Post-Adjust')

    ax.legend()

plt.tight_layout()
plt.show()
