from scipy.stats import ttest_ind_from_stats

models = ["ResNet152", "Wide ResNet101", "VGG", "ResNet50", "ResNet18", "EfficientNet", "RegNet", "ConvNext"]
fpr_means_white = [0.8765, 0.54425, 0.8055, 0.5255, 0.7615, 0.73907, 0.8711, 0.5068]
fpr_stds_white = [0.08847, 0.10654, 0.06440, 0.09277, 0.08774, 0.09020, 0.08313, 0.12391]
tpr_means_white = [0.8614, 0.62345, 0.9010, 0.6772, 0.7652, 0.81995, 0.82334, 0.67951]
tpr_stds_white = [0.03166, 0.04687, 0.02581, 0.04091, 0.03862, 0.03885, 0.03098, 0.03624]

fpr_means_non_white = [0.8032, 0.54165, 0.7496, 0.3792, 0.6493, 0.7519, 0.8129, 0.4247]
fpr_stds_non_white = [0.13768, 0.12543, 0.12078, 0.11891, 0.10568, 0.11883, 0.08817, 0.12211]
tpr_means_non_white = [0.8339, 0.6161, 0.8607, 0.6464, 0.7396, 0.8053, 0.7901, 0.6696]
tpr_stds_non_white = [0.05527, 0.09524, 0.05875, 0.09321, 0.08348, 0.06672, 0.08455, 0.10020]
# Correcting the code to use ttest_ind_from_stats for the T-tests
t_test_results_fpr = []
t_test_results_tpr = []

# Sample size for each group
n_white = 32
n_non_white = 9

for i in range(len(models)):
    # T-test for FPR
    t_stat_fpr, p_value_fpr = ttest_ind_from_stats(mean1=fpr_means_white[i], std1=fpr_stds_white[i], nobs1=n_white,
                                                   mean2=fpr_means_non_white[i], std2=fpr_stds_non_white[i], nobs2=n_non_white)
    # T-test for TPR
    t_stat_tpr, p_value_tpr = ttest_ind_from_stats(mean1=tpr_means_white[i], std1=tpr_stds_white[i], nobs1=n_white,
                                                   mean2=tpr_means_non_white[i], std2=tpr_stds_non_white[i], nobs2=n_non_white)

    t_test_results_fpr.append((models[i], t_stat_fpr, p_value_fpr))
    t_test_results_tpr.append((models[i], t_stat_tpr, p_value_tpr))

t_test_results_fpr, t_test_results_tpr