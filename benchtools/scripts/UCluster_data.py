'''
Script to convert the results of UCluster into the format needed for the benchmarking with run.py
'''

import h5py
import pickle
import argparse
import numpy as np
import pandas as pd
from scripts.run import classifier

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='Path for the file with the data to transform')
parser.add_argument('--box', type=int, default=1, help='Black Box number, ignored if RD dataset [default: 1]')
parser.add_argument('--RD',  type=bool, default=True ,help='If is from the RD dataser [default: False')

flags = parser.parse_args()

DATA = flags.path  #rd_dist_full.h5
NBOX = flags.box
RD = flags.RD

# Setting the variables for the classfier object
name = 'UCluster'
score = np.array(h5py.File(DATA, 'r')['distances'])
pred = np.array(h5py.File(DATA, 'r')['pid'])
label = np.array(h5py.File(DATA, 'r')['label'])

# To calculate the score for a clustering algorithm
norm = np.linalg.norm(score[:,1])

# Creating classifier object
clf = classifier(name, 1-score[:,1]/norm, pred, label)

# Saving
if RD is True: filename = '{}_{}.sav'.format(name,RD)
else: filename = '{}_BB{}.sav'.format(name,NBOX)

pickle.dump(clf, open('../../data/models/{}'.format(filename), 'wb'))