'''
Script to convert the results of UCluster into the format needed for the benchmarking with run.py
'''

import os
import h5py
import pickle
import argparse
import numpy as np
import pandas as pd
from benchtools.src.metrictools import classifier

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../../data', help='Path for the file with the data to transform [default: ../data]')
parser.add_argument('--box', type=int, default=1, help='Black Box number, ignored if RD dataset [default: 1]')
parser.add_argument('--RD',  default=False, action='store_true',help='Use RD data set [default: False]')

flags = parser.parse_args()

DATA = flags.path  #rd_dist_full.h5
NBOX = flags.box
RD = flags.RD

if RD:
    sample = 'full_data_full_training_RD.h5'

else:
    sample = 'full_data_full_training_BB{}.h5'.format(NBOX)

# Setting the variables for the classfier object
name = 'UCluster'
score = np.array(h5py.File(os.path.join(DATA, sample), 'r')['distances'])
pred = np.array(h5py.File(os.path.join(DATA, sample), 'r')['pid'])
label = np.array(h5py.File(os.path.join(DATA, sample), 'r')['label'])

# To calculate the score for a clustering algorithm
norm = np.linalg.norm(score[:,1])

# Creating classifier object
clf = classifier(name, 1-score[:,1]/norm, pred, label)

# Saving
if RD is True: filename = '{}_RD.sav'.format(name)
else: filename = '{}_BB{}.sav'.format(name,NBOX)

pickle.dump(clf, open('../../data/{}'.format(filename), 'wb'))