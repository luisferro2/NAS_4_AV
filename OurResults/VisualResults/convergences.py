""" File with functionality to generate convergence visualizations."""

import os
import matplotlib.pyplot as plt
import pandas as pd
from constants import debug, fol2d

plt.style.use('ggplot')

bp = os.path.join('surrogates', 'float64', 'sequential', 'STRIX', 'GA')
def convergence_plot_medians(folder, feature_type):
    i = 0
    conv_df = pd.DataFrame()
    con_f = os.path.join(bp, folder, feature_type)
    
    for file in os.listdir(con_f):
        if file == '.DS_Store':
            continue

        if 'convergence' in file:
            curr_con_df = pd.read_csv(os.path.join(con_f, file))
            conv_df[f'fitness{i}'] = curr_con_df['fitness']
            i += 1

    conv_dft = conv_df.transpose()
    medians = conv_dft.median()

    plt.figure(figsize=(7, 5))
    plt.boxplot(conv_dft, vert=True, positions=conv_df.index + 1, 
                whiskerprops=dict(linestyle='--', color='orange'),
                boxprops=dict(color='orange'),
                capprops=dict(color='orange'),  #whisker caps
                flierprops=dict(markeredgecolor='orange'))
    plt.plot(conv_df.index + 1, medians, color='g', label='Median')
    plt.xlabel('Generation', fontsize=14)
    plt.xticks(list(range(5, 55))[::5], list(range(5, 55))[::5])
    plt.ylabel('Fitness', fontsize=14)
    plt.legend(fontsize=14)
    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join('figures', f'convergence{folder}{feature_type}.png'), dpi=300)

for dfolder in os.listdir(bp):
    if dfolder == '.DS_Store':
        continue
    for feature_type in ['synBetter', 'sem']:
        convergence_plot_medians(dfolder, feature_type)
