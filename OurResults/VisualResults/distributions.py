""" File with functinoality to generate box-whiskers plots comparing performance 
samples from different methods on multiple datasets."""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from constants import debug, f2d, f2d_svm, our_bests_sequential, svm_bests, baselines, \
    pan_bests

import scipy.stats as stats
plt.style.use('ggplot')

# ------------------------------------------------------------------------------
# Section to compare partial training results between BERT features and glove word 
# embeddings.

base_folder_sequential = os.path.join('surrogates', 'float64', 
                                             'sequential', 'STRIX', 'GA')

# Word embeddings pretraining 2013 folder
wembs_pre_2013f = os.path.join(base_folder_sequential, '2013', 'sem')
cembs_pre_2013f = os.path.join(base_folder_sequential, '2013', 'semBert')

wembs_db_2013 = []
for file in os.listdir(wembs_pre_2013f):
    if 'convergence' in file:
        continue
    # Current pretraining results
    curr_prers = pd.read_csv(os.path.join(wembs_pre_2013f, file))
    wembs_db_2013 += curr_prers['combined accuracy'].to_list()[:24]

cembs_db_2013 = []
for file in os.listdir(cembs_pre_2013f):
    if 'convergence' in file:
        continue
    # Current pretraining results
    curr_prers = pd.read_csv(os.path.join(cembs_pre_2013f, file))
    cembs_db_2013 += curr_prers['combined accuracy'].to_list()[:24]

plt.boxplot([wembs_db_2013, cembs_db_2013],
            labels=['GloVe word embeddings',
                    'BERT embeddings'])
plt.ylabel('Fitness')
plt.savefig(os.path.join('figures', 'distributionWembsBert.png'), dpi=200)

u_statistic1, p_value1 = stats.mannwhitneyu(wembs_db_2013, cembs_db_2013)
debug(p_value1, 'mann whitney u test between bert and glove')

# ------------------------------------------------------------------------------

bavg = np.average(list(baselines.values()))
bstd = np.std(list(baselines.values()))

clabs = ['NAS for AV \nCLEF-PAN \n2013', 'Submissions \nCLEF-PAN \n2013',
        'NAS for AV \nCLEF-PAN \n2014 Essay', 'Submissions \nCLEF-PAN \n2014 Essay',
        'NAS for AV \nCLEF-PAN \n2014 Novel', 'Submissions \nCLEF-PAN \n2014 Novel',
        'NAS for AV \nCLEF-PAN \n2015', 'Submissions \nCLEF-PAN \n2015',
        'NAS for AV \nCLEF-PAN \n2020', 'Submissions \nCLEF-PAN \n2020']

plt.figure(figsize=(15, 7))
plt.boxplot([our_bests_sequential['CLEF-PAN 2013'], pan_bests['CLEF-PAN 2013'],
             our_bests_sequential['CLEF-PAN 2014 Essay'], pan_bests['CLEF-PAN 2014 Essay'],
             our_bests_sequential['CLEF-PAN 2014 Novel'], pan_bests['CLEF-PAN 2014 Novel'],
             our_bests_sequential['CLEF-PAN 2015'], pan_bests['CLEF-PAN 2015'],
             our_bests_sequential['CLEF-PAN 2020'], pan_bests['CLEF-PAN 2020']], 
             labels=clabs,
             widths=0.25)
#plt.title('Methods to solve AV on selected corpora.')
plt.axhline(bavg, c='b', marker=0, label='Baseline average')

ax = plt.gca()  # Get current axis.

# Change font size for x-axis labels
ax.set_xticklabels(clabs, fontsize=14)
# Change font size for y-axis labels (tick labels)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylabel("Performance (F1 for CLEF-PAN 2013, AUC for rest)", fontsize=14)  # Y-axis label

for label in ax.get_xticklabels():
    if 'nas for av' in label.get_text().lower():
        label.set_color('brown')
    else:
        label.set_color('slateblue')

plt.legend(fontsize=14)
plt.savefig(os.path.join('figures', 'distributionsGood.png'), dpi=200)

plt.clf()

clabs = ['SVM \nCLEF-PAN \n2013', 'Submissions \nCLEF-PAN \n2013',
        'SVM \nCLEF-PAN \n2014 Essay', 'Submissions \nCLEF-PAN \n2014 Essay',
        'SVM \nCLEF-PAN \n2014 Novel', 'Submissions \nCLEF-PAN \n2014 Novel',
        'SVM \nCLEF-PAN \n2015', 'Submissions \nCLEF-PAN \n2015',
        'SVM \nCLEF-PAN \n2020', 'Submissions \nCLEF-PAN \n2020']
plt.figure(figsize=(15, 7))
plt.boxplot([svm_bests['CLEF-PAN 2013'], pan_bests['CLEF-PAN 2013'],
             svm_bests['CLEF-PAN 2014 Essay'], pan_bests['CLEF-PAN 2014 Essay'],
             svm_bests['CLEF-PAN 2014 Novel'], pan_bests['CLEF-PAN 2014 Novel'],
             svm_bests['CLEF-PAN 2015'], pan_bests['CLEF-PAN 2015'],
             svm_bests['CLEF-PAN 2020'], pan_bests['CLEF-PAN 2020']], 
             labels=clabs,
             widths=0.25)
#plt.title('Methods to solve AV on selected corpora.')
#plt.axhline(0.5, c='y')
plt.axhline(bavg, c='b', marker=0, label='Baseline average')

ax = plt.gca()  # Get current axis.
# Change font size for x-axis labels
ax.set_xticklabels(clabs, fontsize=14)
# Change font size for y-axis labels (tick labels)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylabel("Performance (F1 for CLEF-PAN 2013, AUC for rest)", fontsize=14)  # Y-axis label
plt.legend(fontsize=14)
#plt.show()
plt.savefig(os.path.join('figures', 'distributionsBad.png'), dpi=200)


for dataset in our_bests_sequential:
    if dataset == '2014Essay1' or dataset == '2014Novel1' or dataset == 'final':
        continue
    clabs = ['SVM ', 'Submissions', 
                        'NAS for AV']
    plt.figure(figsize=(7, 7))
    plt.boxplot([svm_bests[dataset], pan_bests[dataset], our_bests_sequential[dataset],],
                labels=clabs,
                widths=0.25)
    #plt.title('Methods to solve AV on selected corpus.')
    
    plt.axhline(baselines[dataset], c='b', marker=0, label='Baseline')
    plt.axhline(max(pan_bests[dataset]), c='g', marker=0, label='Best result')

    ax = plt.gca()  # Get current axis.
    # Change font size for x-axis labels
    ax.set_xticklabels(clabs, fontsize=16)
    # Change font size for y-axis labels (tick labels)
    ax.tick_params(axis='y', labelsize=16)

    if dataset == 'CLEF-PAN 2013':
        ax.set_ylabel("F1 Score", fontsize=16)  # Y-axis label
    else:
        ax.set_ylabel("AUC", fontsize=16)  # Y-axis label
    plt.legend(fontsize=14)
    #plt.show()
    plt.savefig(os.path.join('figures', f'distributions{dataset}.png'), dpi=200)


