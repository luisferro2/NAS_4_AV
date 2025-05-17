""" File with functionality to analyze the patterns of substructures in the DNN
architectures."""

import os
import itertools
import pandas as pd
from constants import debug

base_folder_sequential = os.path.join('surrogates', 'float64', 
                                             'sequential', 'STRIX', 'GA')


# Pattern finding
#----------------------------------------------------------------------------
def decode(encoding):
    return [int(num) for num in encoding.split(',')]

int_to_class = {0: 0,
                1: 1,
                2: 1,
                3: 1,
                4: 2,
                5: 2,
                6: 2,
                7: 3,
                8: 3,
                9: 3,
                10: 3,
                11: 3,
                12: 3,
                13: 3,
                14: 3,
                15: 3,
                16: 3,
                17: 3,
                18: 3,
                19: 3,
                # 20: 3,
                20: 4,
                21: 4,
                22: 4,
                23: 4}


def compute_prefix(p):
    m = len(p)
    pi = [0] * m
    
    k = 0
    for q in range(2, m + 1):
        # debug(q, 'q')
        while k > 0 and p[k] != p[q - 1]:
            k = pi[k - 1]
        if p[k] == p[q - 1]:
            k += 1
        pi[q - 1] = k
        # debug(pi[q - 1], 'pi[q - 1]')
    return pi
 
def kmp_matcher(t, p):
    matches_starts = []
    n_matches = 0
    n = len(t)
    m = len(p)
    pi = compute_prefix(p)
    # debug(pi, 'pi')
    q = 0
    for i in range(1, n + 1):
        while q > 0 and p[q] != t[i - 1]:
            q = pi[q - 1]
        if p[q] == t[i - 1]:
            q += 1
        if q == m:
            # debug(i - m, 'pattern found')
            matches_starts += [i - m]
            n_matches += 1
            q = pi[q - 1]
    return n_matches, matches_starts

# b = [0, 0, 1]
# a = [3, 3, 4, 4, 0, 4, 3, 3, 4, 2, 3, 3, 3, 3, 3]
# debug(kmp_matcher(a, b), 'kmp matcher')

syn_best_classes = []
sem_best_classes = []

for folder in sorted(os.listdir(base_folder_sequential)):
    if folder == '.DS_Store':
        continue

    # Syntax full training df.
    syntax_ftdf = pd.read_csv(os.path.join(base_folder_sequential, folder, 'surrogateFTDFSyntax.csv'))

    syntax_ftdf.sort_values('test auc', inplace=True, ascending=False)
    syntax_ftdf = syntax_ftdf.loc[syntax_ftdf['training accuracy'] > 0.5]

    # Syntax encodings string
    syn_enc_str = syntax_ftdf['encoding'].unique()[:5]

    # Semantic full training df.
    semantic_ftdf = pd.read_csv(os.path.join(base_folder_sequential, folder, 'surrogateFTDFSemantic.csv'))

    semantic_ftdf.sort_values('test auc', inplace=True, ascending=False)
    semantic_ftdf = semantic_ftdf.loc[semantic_ftdf['training accuracy'] > 0.5]

    # Semantic encodings string
    sem_enc_str = semantic_ftdf['encoding'].unique()[:5]

    syn_enc_str = [decode(enc) for enc in syn_enc_str]
    sem_enc_str = [decode(enc) for enc in sem_enc_str]

    syn_best_classes += [[int_to_class[num] for num in enc] for enc in syn_enc_str]
    sem_best_classes += [[int_to_class[num] for num in enc] for enc in sem_enc_str]

debug(syn_best_classes, 'syn best classes')
debug(sem_best_classes, 'sem best classes')

# Modify this value for different-sized patterns
patt_len = 8

# We calculate all possible patterns of a given length.
possible_patts = [p for p in itertools.product(list(range(5)), repeat=patt_len)]

patt_cnt = {}
for possible_patt in possible_patts:
    # debug(possible_patt, 'possible pattern')
    patt_cnt[possible_patt] = 0
    for curr_best in syn_best_classes + sem_best_classes:
        # debug(curr_best, 'curr best')
        patt_cnt[possible_patt] += kmp_matcher(curr_best, possible_patt)[0]
    
patt_cnt = {k: v for k, v in sorted(patt_cnt.items(), key=lambda item: item[1])}
patt_cnt_good = {k: patt_cnt[k] for k in patt_cnt if patt_cnt[k] > 0 and patt_cnt[k] < 3}


debug(patt_cnt_good, 'pattern counter')

quit()

# Patterns with most frequency across top DNNs.
# Length 3: (2, 3, 4): 8 = activation- linear - convolution
# Length 4: (3, 2, 1, 3): 7 = linear - activation - dropout - linear
# Length 5: (3, 2, 3, 4, 3): 5 = linear - activation - linear - conv1d - linear
# Length 6: (3, 2, 3, 4, 3, 3): 3 = linear - activation - linear - conv1d - linear - linear

# # Patterns with least frequency across top DNNs.
# Length 7: (3, 1, 1, 4, 3, 2, 2): 1 = linear - dropout - dropout - conv1d - linear - activation - activation


# This section below is to find all the instances where the pattern is found.
chosen_pattern = [3, 2, 1, 2, 3, 1, 3]

def fill_zeroes(pattern):
    n_slots = len(pattern) - 1

    possibilities = []
    for i in range(n_slots):
        possibilities += [pattern[:i + 1] + [0] + pattern[i + 1:]]
    return possibilities

for pattern in [chosen_pattern, fill_zeroes(chosen_pattern)]:
    for curr_ensembles in os.listdir('ensemblesSequential'):
        if curr_ensembles == '.DS_Store' or curr_ensembles == 'ensemblesDF2014Essay1.csv' or\
                curr_ensembles == 'ensemblesDF2014Novel1.csv':
            continue
        # debug(curr_ensembles, 'current ensembles name')
        curr_ensemblesdf = pd.read_csv(os.path.join('ensembles', curr_ensembles))
        
        for i in range(5):
            # Current syntax model.
            curr_synm = decode(curr_ensemblesdf.loc[i]['syntax model'])
            # Current syntax model classes.
            curr_synmc = [int_to_class[num] for num in curr_synm]

            # Current semantic model.
            curr_semm = decode(curr_ensemblesdf.loc[i]['semantic model'])
            # Current semantic model classes.
            curr_semmc = [int_to_class[num] for num in curr_semm]

            # Current pattern count.
            curr_pc_syn, curr_kmp_idxs_syn = kmp_matcher(curr_synmc, pattern)
            curr_pc_sem, curr_kmp_idxs_sem = kmp_matcher(curr_semmc, pattern)

            if curr_pc_syn > 0:
                debug(curr_ensembles + ' ' + f'{i}, {curr_pc_syn}, {curr_kmp_idxs_syn}' , f'pattern {pattern} found in syntax top, times, position')
            if curr_pc_sem > 0:
                debug(curr_ensembles + ' ' + f'{i}, {curr_pc_sem}, {curr_kmp_idxs_sem}', f'pattern {pattern} found in semantic top, times, position')

