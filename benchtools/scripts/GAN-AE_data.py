'''
Script to convert the results of GAN-AE into the format needed for the benchmarking with run.py
'''
import os
import h5py
import pickle
import argparse
import numpy as np
import pandas as pd
from scripts.run import classifier
from benchtools.src.metrictools import optimal_threshold

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='..\data', help='Path for the file with the data to transform [default: ..\data]')
parser.add_argument('--box', type=int, default=1, help='Black Box number, ignored if RD dataset [default: 1]')
parser.add_argument('--RD',  type=bool, default=True ,help='If is from the RD dataser [default: False')

flags = parser.parse_args()

DATA = flags.path 
NBOX = flags.box
RD = flags.RD

if RD:
    sample = 'RnD_distances.h5'
else:
    sample = 'BB{}_distances.h5'.format(NBOX)

# Reading the distances
dist_bkg = pd.DataFrame(np.array(h5py.File(os.path.join(DATA,sample), 'r')['bkg']), columns=['y_score'])
dist_sig = pd.DataFrame(np.array(h5py.File(os.path.join(DATA,sample), 'r')['sig1']), columns=['y_score'])

# Adding labels, 0 for background and 1 for signal
dist_bkg['label']=0
dist_sig['label']=1

# Getting together signal with background
df_sig = pd.concat([dist_bkg, dist_sig])

# Setting the variables for the classfier object
name = 'GAN-AE'
label = np.array(df_sig.loc[:,'label'])
score = np.array(df_sig.loc[:,'y_score'])

# Calculating predictions with the better threshold
threshold = optimal_threshold(label, score)
df_sig['y_pred']=(df_sig['y_score'] >= threshold).astype(float)

# Setting the prediction variable
pred = df_sig.loc[:,'y_pred']

# Creating classifier object
clf = classifier(name, score, pred, label)

# Saving
if RD: 
    filename = '{}_RD.sav'.format(name)
else: 
    filename = '{}_BB{}.sav'.format(name,NBOX)

pickle.dump(clf, open('../../data/{}'.format(filename), 'wb'))