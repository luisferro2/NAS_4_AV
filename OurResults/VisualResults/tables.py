""" File with functionality to generate latex table showing the performance
measurement results obtained by our approach, by the CLEF-PAN submissions, and the SVM
approach."""


import os
import pandas as pd
import numpy as np

from constants import f2d, d2f_svm, debug

debug('FOR TABLE ON PAN DATASETS', '')


for curr_ensembles in sorted(os.listdir('ensemblesSequential')):
    if curr_ensembles == '.DS_Store' or curr_ensembles == 'ensemblesDFFinal.csv':
        continue
    # debug(curr_ensembles, 'current ensembles name')
    curr_dataset = f2d[curr_ensembles]
    print(rf'''\hline
    \hline
    \multicolumn{{7}}{{c}}{{\textbf{{{curr_dataset}}}}}\\
    \hline
    \textbf{{Top}} & \multirow{{2}}{{*}}{{\textbf{{Accuracy}}}} & \multirow{{2}}{{*}}{{\textbf{{Precision}}}} & \multirow{{2}}{{*}}{{\textbf{{Recall}}}} & \multirow{{2}}{{*}}{{\textbf{{F1 score}}}} & \multirow{{2}}{{*}}{{\textbf{{AUC}}}} & \multirow{{2}}{{*}}{{\textbf{{Method used}}}}\\
    \textbf{{Experiments}}& & & & & & \\
    \hline''')
    curr_ensemblesdf = pd.read_csv(os.path.join('ensembles', curr_ensembles))
    for i in range(3):
        curr_ensemble = curr_ensemblesdf.loc[i]
        if i == 0:
            print(rf'''{i + 1} & {curr_ensemble['test accuracy']:.4f} & {curr_ensemble['test precision']:.4f} & {curr_ensemble['test recall']:.4f} & {curr_ensemble['test fscore']:.4f} & {curr_ensemble['test auc']:.4f} & \multirow{{5}}{{*}}{{NAS for AV}}\\''')
        else:
            print(rf'''{i + 1} & {curr_ensemble['test accuracy']:.4f} & {curr_ensemble['test precision']:.4f} & {curr_ensemble['test recall']:.4f} & {curr_ensemble['test fscore']:.4f} & {curr_ensemble['test auc']:.4f} & \\''')
        print('\cline{1-6}')
    print(rf"Avg. & {np.average(curr_ensemblesdf[:3]['test accuracy']):.4f} & {np.average(curr_ensemblesdf[:3]['test precision']):.4f} & {np.average(curr_ensemblesdf[:3]['test recall']):.4f} & {np.average(curr_ensemblesdf[:3]['test fscore']):.4f} & {np.average(curr_ensemblesdf[:3]['test auc']):.4f} &\\")
    print('\cline{1-6}')
    print(rf"\tiny Std. Dev. & \tiny {np.std(curr_ensemblesdf[:3]['test accuracy']):.4f} & \tiny {np.std(curr_ensemblesdf[:3]['test precision']):.4f} & \tiny {np.std(curr_ensemblesdf[:3]['test recall']):.4f} & \tiny {np.std(curr_ensemblesdf[:3]['test fscore']):.4f} & \tiny {np.std(curr_ensemblesdf[:3]['test auc']):.4f} &\\")
    print(rf"\hline")
    
    curr_svm = d2f_svm[curr_dataset]
    curr_svm_df = pd.read_csv(os.path.join('svm_results', curr_svm))
    if curr_svm == 'svm_2013_df.csv':
        curr_svm_df = curr_svm_df.sort_values(['test fscore', 'test auc', 'test accuracy'], ascending=False)
    else:
        curr_svm_df = curr_svm_df.sort_values(['test auc', 'test accuracy', 'test fscore'], ascending=False)

    curr_svm_row = curr_svm_df.iloc[0]
    print(rf'\multirow{{3}}{{*}}{{N/A}} &&&&&& CLEF-PAN Best\\')
    print(rf'\cline{{2-7}}')
    print(rf' &&&&&& CLEF-PAN Baseline\\')
    print(rf'\cline{{2-7}}')
    print(rf'& {curr_svm_row['test accuracy']:.4f} & {curr_svm_row['test precision']:.4f} & {curr_svm_row['test recall']:.4f} & {curr_svm_row['test fscore']:.4f} & {curr_svm_row['test auc']:.4f} & SVM\\')
print(r'\hline')
print(r'\hline')

