""" File with the functionality of carrying out the Mann-Whitney U tests when 
comparing performance measurement samples between our approach vs. the SVM, and our 
approach vs. the CLEF-PAN submissions."""

import scipy.stats as stats
from constants import our_bests_sequential, pan_bests, svm_bests


def debug(thing, title):
    print('------------------------------------')
    print(title)
    print(thing)
    print('------------------------------------')

# P-values
# Ours vs PAN
# SVM vs PAN
# Ours vs SVM

debug('FOR TABLE P VALUES mann whitney u test', '')
print('\hline')
print('\hline')
print(r'Dataset & Samples compared & p value & Significant difference\\')
print('\hline')
for ind, dataset in enumerate(pan_bests):
    # _, p_value1rs = stats.ranksums(our_bests[dataset], pan_bests[dataset])

    u_statistic2, p_value2 = stats.mannwhitneyu(our_bests_sequential[dataset], svm_bests[dataset])

    
    if ind == 0:
        print(rf"{dataset} & \multirow{{5}}{{*}}{{NAS for AV vs. SVM}} & {p_value2:.4f} & {'True' if p_value2 < 0.05 else 'False'}\\")
    else:
        print(rf"{dataset} &  & {p_value2:.4f} & {'True' if p_value2 < 0.05 else 'False'}\\")
    print('\cline{1-1}')
    print('\cline{3-4}')
print('\hline')

for ind, dataset in enumerate(pan_bests):
    # _, p_value1rs = stats.ranksums(our_bests[dataset], pan_bests[dataset])

    u_statistic1, p_value1 = stats.mannwhitneyu(our_bests_sequential[dataset], pan_bests[dataset])
    
    if ind == 0:
        print(rf"{dataset} & \multirow{{5}}{{*}}{{NAS for AV vs. Submissions}} & {p_value1:.4f} & {'True' if p_value1 < 0.05 else 'False'}\\")
    else:
        print(rf"{dataset} &  & {p_value1:.4f} & {'True' if p_value1 < 0.05 else 'False'}\\")
    print('\cline{1-1}')
    print('\cline{3-4}')
print('\hline')
print('\hline')

statistic, p_valuef = stats.friedmanchisquare(*our_bests_sequential.values())

print(rf'{p_valuef}')

