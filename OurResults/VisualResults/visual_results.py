''' File with functionality to generate the individual visual representation of the top DNNs
in terms of their performance after full training.'''
import itertools

import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import os
import seaborn as sns
import numpy as np
import networkx as nx
import statistics
from netgraph import Graph

from constants import f2d, debug, fol2d


# Individuals visualizations.

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

colors = ['#000000', '#4B1932', '#649632', '#4B96E1', '#FFC864']

int_to_color = {0: colors[0],
                1: colors[1],
                2: colors[1],
                3: colors[1],
                4: colors[2],
                5: colors[2],
                6: colors[2],
                7: colors[3],
                8: colors[3],
                9: colors[3],
                10: colors[3],
                11: colors[3],
                12: colors[3],
                13: colors[3],
                14: colors[3],
                15: colors[3],
                16: colors[3],
                17: colors[3],
                18: colors[3],
                19: colors[3],
                # 20: colors[3],
                20: colors[4],
                21: colors[4],
                22: colors[4],
                23: colors[4]}

markers = ['o', 'v', '8', 's', 'p']

int_to_marker = {0: markers[0],
                1: markers[1],
                2: markers[1],
                3: markers[1],
                4: markers[2],
                5: markers[2],
                6: markers[2],
                7: markers[3],
                8: markers[3],
                9: markers[3],
                10: markers[3],
                11: markers[3],
                12: markers[3],
                13: markers[3],
                14: markers[3],
                15: markers[3],
                16: markers[3],
                17: markers[3],
                18: markers[3],
                19: markers[3],
                # 20: markers[3],
                20: markers[4],
                21: markers[4],
                22: markers[4],
                23: markers[4]}

def display_encoding(encodings, dataset):
    encodings_syn, encodings_sem = encodings

    # Map integers to strings
    mapped_nodes_syn_all = [[int_to_mod[i] for i in encoding_syn] for encoding_syn in encodings_syn]
    mapped_nodes_sem_all = [[int_to_mod[i] for i in encoding_sem] for encoding_sem in encodings_sem]
    debug(mapped_nodes_syn_all, 'mapped nodes syn all')

    # Create two parallel paths
    paths_syn = [[node + '_' + str(i + 1) for i, node in enumerate(mapped_nodes_syn)] \
                 for mapped_nodes_syn in mapped_nodes_syn_all]
    paths_sem = [[node + '_' + str(i + 1) for i, node in enumerate(mapped_nodes_sem)] \
                 for mapped_nodes_sem in mapped_nodes_sem_all]
    debug(paths_syn, 'paths syn')

    colors_syn = [{paths_syn[j][ind]: int_to_color[node] for ind, node in enumerate(encoding_syn)} \
        for j, encoding_syn in enumerate(encodings_syn)]
    
    colors_sem = [{paths_sem[j][ind]: int_to_color[node] for ind, node in enumerate(encoding_sem)} \
        for j, encoding_sem in enumerate(encodings_sem)]
    debug(colors_syn, 'colors syn')

    markers_syn = [{paths_syn[j][ind]: int_to_marker[node] for ind, node in enumerate(encoding_syn)} \
        for j, encoding_syn in enumerate(encodings_syn)]
    
    markers_sem = [{paths_sem[j][ind]: int_to_marker[node] for ind, node in enumerate(encoding_sem)} \
        for j, encoding_sem in enumerate(encodings_sem)]
    debug(markers_syn, 'markers syn')

    plt.clf()
    plt.figure(figsize=(4, 8))
    # plt.subplots(2, 5)
    # plt.suptitle(f"Syntax (left) \n and semantic (right)\n for ensemble \n{dataset}")
    fontsize = 5
    nodesize=40
    for i in range(5):
        # Visualization
        G1 = nx.DiGraph()
        path_syn = paths_syn[i]
        # Add edges for the first path
        for j in range(len(path_syn) - 1):
            G1.add_edge(path_syn[j], path_syn[j + 1])


        ax = plt.subplot(2, 5, i + 1)
        ax.set_title(i + 1)
        # My linspace
        mls = np.linspace(0, 8, 15)

        g1 = Graph(G1,
            node_layout = {path_syn[0]: (i, mls[14]),
                            path_syn[1]: (i, mls[13]),
                            path_syn[2]: (i, mls[12]),
                            path_syn[3]: (i, mls[11]),
                            path_syn[4]: (i, mls[10]),
                            path_syn[5]: (i, mls[9]),
                            path_syn[6]: (i, mls[8]),
                            path_syn[7]: (i, mls[7]),
                            path_syn[8]: (i, mls[6]),
                            path_syn[9]: (i, mls[5]),
                            path_syn[10]: (i, mls[4]),
                            path_syn[11]: (i, mls[3]),
                            path_syn[12]: (i, mls[2]),
                            path_syn[13]: (i, mls[1]),
                            path_syn[14]: (i, mls[0])},
            node_size = nodesize,
            node_color = colors_syn[i],
            node_labels = {name: name for name in path_syn},
            node_label_fontdict = {'backgroundcolor' : (0.75, 0.75, 0.75, 0.75)},#'lightgray'},
            node_shape = markers_syn[i],
            arrows=True,
            )

        for node, label in g1.node_label_artists.items():
            label.set_fontsize(fontsize)



        G2 = nx.DiGraph()

        path_sem = paths_sem[i]
        # Add edges for the second path
        for j in range(len(path_sem) - 1):
            G2.add_edge(path_sem[j], path_sem[j + 1])


        ax2 = plt.subplot(2, 5, 6 + i)
        ax2.set_title(i + 1)
        g2 = Graph(G2,
              node_layout = {path_sem[0]: (1, mls[14]),
                             path_sem[1]: (1, mls[13]),
                             path_sem[2]: (1, mls[12]),
                             path_sem[3]: (1, mls[11]),
                             path_sem[4]: (1, mls[10]),
                             path_sem[5]: (1, mls[9]),
                             path_sem[6]: (1, mls[8]),
                             path_sem[7]: (1, mls[7]),
                             path_sem[8]: (1, mls[6]),
                             path_sem[9]: (1, mls[5]),
                             path_sem[10]: (1, mls[4]),
                             path_sem[11]: (1, mls[3]),
                             path_sem[12]: (1, mls[2]),
                             path_sem[13]: (1, mls[1]),
                             path_sem[14]: (1, mls[0])},
              node_size = nodesize,
              node_color = colors_sem[i],
              node_labels = {name: name for name in path_sem},
              node_label_fontdict = {'backgroundcolor' : (0.75, 0.75, 0.75, 0.75)},
              node_shape = markers_sem[i],
              arrows=True,
            )

        for node, label in g2.node_label_artists.items():
            label.set_fontsize(fontsize)
    
    # Add titles and grid for clarity
    
    # plt.tight_layout(pad=-4, w_pad=-8, h_pad=-10)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join('figures', f'architecture_{dataset}.png'.replace('\n', '')), dpi=200)

def str2list(encoding):
    return [int(num) for num in encoding.split(',')]

# base path
bp = os.path.join('surrogates', 'float64', 'sequential', 'STRIX', 'GA')

for folder in sorted(os.listdir(bp)):
    if folder == '.DS_Store':
        continue

    # Syntax full training df.
    syntax_ftdf = pd.read_csv(os.path.join(bp, folder, 'surrogateFTDFSyntax.csv'))

    syntax_ftdf.sort_values('test auc', inplace=True, ascending=False)
    syntax_ftdf = syntax_ftdf.loc[syntax_ftdf['training accuracy'] > 0.5]

    # Syntax encodings string
    syn_enc_str = syntax_ftdf['encoding'].unique()[:5]

    # Semantic full training df.
    semantic_ftdf = pd.read_csv(os.path.join(bp, folder, 'surrogateFTDFSemantic.csv'))

    semantic_ftdf.sort_values('test auc', inplace=True, ascending=False)
    semantic_ftdf = semantic_ftdf.loc[semantic_ftdf['training accuracy'] > 0.5]

    # Semantic encodings string
    sem_enc_str = semantic_ftdf['encoding'].unique()[:5]

    syn_enc_str = [str2list(enc) for enc in syn_enc_str]
    sem_enc_str = [str2list(enc) for enc in sem_enc_str]

    debug(syn_enc_str, 'syntax encoding strings')
    debug(sem_enc_str, 'semantic encoding strings')

    display_encoding([syn_enc_str, sem_enc_str], fol2d[folder])







