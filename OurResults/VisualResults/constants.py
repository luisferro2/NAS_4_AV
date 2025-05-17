""" File with functionality to load the constant performance measurements from 
multiple sources."""

import os
import pandas as pd

def debug(thing, title):
    print('---------------------------------')
    print(title)
    print(thing)
    print('---------------------------------')


# File to dataset.
f2d = {'ensemblesDF2013.csv': 'CLEF-PAN 2013',
 'ensemblesDF2014Essay1.csv': '2014Essay1',
 'ensemblesDF2014Essay2.csv': 'CLEF-PAN 2014 Essay',
 'ensemblesDF2014Novel1.csv': '2014Novel1',
 'ensemblesDF2014Novel2.csv': 'CLEF-PAN 2014 Novel',
 'ensemblesDF2015.csv': 'CLEF-PAN 2015',
 'ensemblesDF2020.csv': 'CLEF-PAN 2020',
 'ensemblesDFFinal.csv': 'final'}


d2f_svm = {
    'CLEF-PAN 2013': 'svm_2013_df.csv',
    'CLEF-PAN 2014 Essay': 'svm_2014Essay2_df.csv',
    'CLEF-PAN 2014 Novel': 'svm_2014Novel2_df.csv',
    'CLEF-PAN 2015': 'svm_2015_df.csv',
    'CLEF-PAN 2020': 'svm_2020_df.csv',
    'final': 'svm_final_df.csv'
}


f2d_svm = {
    'svm_2013_df.csv': 'CLEF-PAN 2013',
    'svm_2014Essay2_df.csv': 'CLEF-PAN 2014 Essay',
    'svm_2014Novel2_df.csv': 'CLEF-PAN 2014 Novel',
    'svm_2015_df.csv': 'CLEF-PAN 2015',
    'svm_2020_df.csv': 'CLEF-PAN 2020',
    'svm_final_df.csv': 'final'
}

fol2d = {
    '2013': 'CLEF-PAN 2013',
    '2014Essay2': 'CLEF-PAN 2014 Essay',
    '2014Novel2': 'CLEF-PAN 2014 Novel',
    '2015': 'CLEF-PAN 2015',
    '2020': 'CLEF-PAN 2020',
}

# included baselines, not included meta classifiers.
pan_bests = {'CLEF-PAN 2013': [0.8, 0.8, 0.767, 0.767, 0.733, 
            0.733, 0.7, 0.691, 0.667, 0.644, 
            0.633, 0.6, 0.6, 0.533, 0.5, 0.5, 0.467, 0.4],
            'CLEF-PAN 2014 Essay': [0.520, 0.518, 0.543, 0.579, 0.549, 0.572, 
                           0.585, 0.629, 0.599, 0.603, 0.595, 0.620, 
                           0.699, 0.723],
            'CLEF-PAN 2014 Novel': [0.453, 0.491, 0.495, 0.510, 0.540, 0.569, 0.597, 0.612,
                           0.657, 0.628, 0.664, 0.750, 0.733, 0.711],
            'CLEF-PAN 2015': [0.5, 0.401, 0.489, 0.537, 0.507, 0.493, 0.517, 0.530,
                     0.578, 0.602, 0.680, 0.654, 0.639, 0.648, 0.763, 0.709, 
                     0.762, 0.738, 0.739, 0.750, 0.811],
            'CLEF-PAN 2020': [0.293, 0.840, 0.696, 0.778, 0.780, 0.859, 0.786, 
                      0.795, 0.874, 0.866, 0.878, 0.939, 0.940, 0.953, 0.969],}


baselines = {'CLEF-PAN 2013': 0.5, 
             'CLEF-PAN 2014 Essay': 0.543, 
             'CLEF-PAN 2014 Novel': 0.453, 
             'CLEF-PAN 2015': 0.639, 
             'CLEF-PAN 2020': 0.780}


our_bests_sequential = {}
for curr_ensembles in os.listdir('ensemblesSequential'):
    if curr_ensembles == '.DS_Store':
        continue
    # debug(curr_ensembles, 'current ensembles name')

    curr_ensemblesdf = pd.read_csv(os.path.join('ensemblesSequential', curr_ensembles))
    # debug(curr_ensemblesdf, 'curr ensembles df')
    if curr_ensembles == 'ensemblesDF2013.csv':
        our_bests_sequential[f2d[curr_ensembles]] = curr_ensemblesdf['test fscore'][:5]
    else:
        our_bests_sequential[f2d[curr_ensembles]] = curr_ensemblesdf['test auc'][:5]

svm_bests = {}
for curr_svm in os.listdir('svm_results'):
    if curr_svm == '.DS_Store':
        continue
    
    debug(curr_svm, 'current svm')
    curr_svm_df = pd.read_csv(os.path.join('svm_results', curr_svm))

    if curr_svm == 'svm_2013_df.csv':
        curr_svm_df = curr_svm_df.sort_values(['test fscore', 'test auc', 'test accuracy'], ascending=False)
    else:
        curr_svm_df = curr_svm_df.sort_values(['test auc', 'test accuracy', 'test fscore'], ascending=False)

    if curr_svm == 'svm_2013_df.csv':
        svm_bests[f2d_svm[curr_svm]] = curr_svm_df['test fscore'][:5]
    else:
        svm_bests[f2d_svm[curr_svm]] = curr_svm_df['test auc'][:5]