""" File with functionality to generate the table analyzing the permutation 
importance of the layer positions in the encoding of the DNNs with
respect to performance, as well as the heatmaps that measure the frequency of 
the layer positions being used as decision nodes in the decision tree regressor."""

import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from constants import debug, fol2d
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance

int_to_mod = {0: 'Identity', 
            1: 'Dropout(0.9)', 
            2: 'Dropout(0.6)', 
            3: 'Dropout(0.3)', 
            4: 'ReLU', 
            5: 'TanH', 
            6: 'Leaky ReLU',
            7: 'Linear(1)', 
            8: 'Linear(2)',
            9: 'Linear(4)',
            10: 'Linear(8)',
            11: 'Linear(16)',
            12: 'Linear(32)',
            13: 'Linear(64)',
            14: 'Linear(128)',
            15: 'Linear(256)',
            16: 'Linear(512)',
            17: 'Linear(1024)',
            18: 'Linear(2048)',
            19: 'Linear(4096)',
            # 20: 'Linear(8192)',
            20: 'Conv1D(2, 2)',
            21: 'Conv1D(2, 8)',
            22: 'Conv1D(8, 2)',
            23: 'Conv1D(8, 8)',}



def str2list(encoding):
    return [int(num) for num in encoding.split(',')]


def count_feature_splits(tree, num_features):
    split_count = np.zeros(num_features)
    for node_id in range(tree.node_count):
        feature_index = tree.feature[node_id]
        # debug(feature_index, 'feature index')
        if feature_index >= 0:  # -2 means leaf node
            split_count[feature_index] += 1
    return split_count

hidden_obs = {}
print('\hline')

syntax_clists = []
semantic_clists = []

base_folder_sequential = os.path.join('surrogates', 'float64', 
                                             'sequential', 'STRIX', 'GA')

for folder in sorted(os.listdir(base_folder_sequential)):
    if folder == '.DS_Store':
        continue
    print('\hline')
    print(rf'\multicolumn{{5}}{{c}}{{\textbf{{{fol2d[folder]}}}}}\\')
    print('\hline')

    print(r'\textbf{Feature used} & \textbf{Most important layer} & \textbf{Models ranking} & \textbf{Values at important layer} & \textbf{Observations}\\')
    print('\hline')
    
    db = pd.read_csv(os.path.join(base_folder_sequential, folder, f'surrogateFTDFSyntax.csv'))

    X = list(db['encoding'].apply(str2list))
    X = np.array(X)

    y = db['test auc'].to_numpy()

    regressor = DecisionTreeRegressor()

    regressor.fit(X, y)
    regressor.score(X, y)

    split_counts = count_feature_splits(regressor.tree_, X.shape[1])
    split_counts = [int(split_count) for split_count in split_counts]
    syntax_clists += [split_counts]

    # str_counts = ""
    # for count in split_counts:
    #     str_counts += rf'{count} &'
    # str_counts = str_counts[:-1]
    # str_counts += rf'\\'
    # print(str_counts)
    
        
    df = pd.DataFrame(X, columns=[f'Layer\_{i + 1}' for i in range(X.shape[1])])
    df['Performance'] = y

    result = permutation_importance(regressor, X, y, n_repeats=20, random_state=42)
    sorted_idx = result.importances_mean.argsort()
    important_layer = np.array(df.columns)[sorted_idx][-1]
    # as index
    important_layer_ai = int(important_layer[important_layer.find('_') + 1:]) - 1
    # debug(important_layer, 'important layer')
    # debug(important_layer_ai, 'important layer as index')

    db.sort_values('test auc', inplace=True, ascending=False)
    db = db.loc[db['training accuracy'] > 0.5]

    # encodings string
    best_enc_str = db['encoding'].unique()[:5]
    # debug(best_sem_enc_str, 'best semantic encodings string')
    best_matrix = pd.DataFrame(list(str2list(enc) for enc in best_enc_str))
    # debug(best_sem_matrix, 'best semantic matrix')
    worst_enc_str = db['encoding'].unique()[-5:]
    worst_matrix = pd.DataFrame(list(str2list(enc) for enc in worst_enc_str))

    top_values = best_matrix[important_layer_ai].values
    bottom_vals = worst_matrix[important_layer_ai].values

    # debug(top_values, 'top values')

    print(rf'\multirow{{2}}{{*}}{{Syntax}} & \multirow{{2}}{{*}}{{{important_layer}}} & Best & {top_values} & a\\')
    print('\cline{3-5}')
    print(rf'Features & & Worst & {bottom_vals} & a\\')
    print('\hline')

    hidden_obs[folder] = [[int_to_mod[num] for num in top_values], [int_to_mod[num] for num in bottom_vals]]
    
    plt.boxplot(df[important_layer])
    plt.savefig(os.path.join('feature_analysis', f'importantLayerValsSyntax{folder}.png'), dpi=200)
    
    # dataset, most important layer position, most common three values, interpretation


    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(df.columns)[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.savefig(os.path.join('feature_analysis', f'{folder}Syntax.png'), dpi=200)
    plt.clf()
    
    db = pd.read_csv(os.path.join(base_folder_sequential, folder, f'surrogateFTDFSemantic.csv'))

    X = list(db['encoding'].apply(str2list))
    X = np.array(X)

    y = db['test auc'].to_numpy()

    regressor = DecisionTreeRegressor()

    regressor.fit(X, y)
    regressor.score(X, y)

    split_counts = count_feature_splits(regressor.tree_, X.shape[1])
    split_counts = [int(split_count) for split_count in split_counts]

    # str_counts = ""
    # for count in split_counts:
    #     str_counts += rf'{count} &'
    # str_counts = str_counts[:-1]
    # str_counts += rf'\\'
    # print(str_counts)
    # print('\hline')

    semantic_clists += [split_counts]

    df = pd.DataFrame(X, columns=[f'Layer\_{i + 1}' for i in range(X.shape[1])])
    df['Performance'] = y

    result = permutation_importance(regressor, X, y, n_repeats=20, random_state=42)
    sorted_idx = result.importances_mean.argsort()
    important_layer = np.array(df.columns)[sorted_idx][-1]
    # as index
    important_layer_ai = int(important_layer[important_layer.find('_') + 1:]) - 1
    # debug(important_layer, 'important layer')
    # debug(important_layer_ai, 'important layer as index')

    db.sort_values('test auc', inplace=True, ascending=False)
    db = db.loc[db['training accuracy'] > 0.5]

    # encodings string
    best_enc_str = db['encoding'].unique()[:5]
    # debug(best_sem_enc_str, 'best semantic encodings string')
    best_matrix = pd.DataFrame(list(str2list(enc) for enc in best_enc_str))
    # debug(best_sem_matrix, 'best semantic matrix')
    worst_enc_str = db['encoding'].unique()[-5:]
    worst_matrix = pd.DataFrame(list(str2list(enc) for enc in worst_enc_str))

    top_values = best_matrix[important_layer_ai].values
    bottom_vals = worst_matrix[important_layer_ai].values

    # debug(top_values, 'top values')

    print(rf'\multirow{{2}}{{*}}{{Semantic}} & \multirow{{2}}{{*}}{{{important_layer}}} & Best & {top_values} & a\\')
    print('\cline{3-5}')
    print(rf'Features & & Worst & {bottom_vals} & a\\')
    print('\hline')

    hidden_obs[folder] += [[int_to_mod[num] for num in top_values], [int_to_mod[num] for num in bottom_vals]]

    plt.boxplot(df[important_layer])
    plt.savefig(os.path.join('feature_analysis', f'importantLayerValsSemantic{folder}.png'), dpi=200)

    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(df.columns)[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.savefig(os.path.join('feature_analysis', f'{folder}Semantic.png'), dpi=200)
    plt.clf()

 # Convert the list of lists to a NumPy array for easier handling

def plot_heatmap(data, name):

    # Create the heatmap
    plt.figure(figsize=(15, 6))  # Adjust figure size as needed

    folders = sorted(os.listdir(base_folder_sequential))[1:]
    d_labels = [fol2d[folder] for folder in folders]

    norm_data = data / data.sum(axis=1)[:, np.newaxis]
    sns.heatmap(norm_data, annot=True, cmap="Blues", fmt=".1%",
            xticklabels=range(1, 16),  # Positions 1 to 15
            yticklabels=d_labels)#number of trees
    
    plt.xlabel("DNN Layer index", fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.ylabel("")
    plt.tight_layout()

    plt.savefig(os.path.join('feature_analysis', name), dpi=200)

plot_heatmap(np.array(syntax_clists), 'syntaxTreeFeatures.png')
plot_heatmap(np.array(semantic_clists), 'semanticTreeFeatures.png')

debug(hidden_obs, 'hidden observations')