'''
Script to convert the results of UCluster into the format needed for the benchmarking with run.py
'''

import os
import h5py
import pickle
import argparse
import numpy as np
import pandas as pd
#from scripts.run import classifier

# Pasting the class here while I don't know how to import it from run
class classifier:
    def __init__(self, name, score, pred, label):
        self.name = name
        self.score = score
        self.pred = pred       
        self.label = label
    
    def precision(self):
        return precision_score(self.label, self.pred)
        
    def recall(self):
        return recall_score(self.label, self.pred)

    def f1_score(self):
        return f1_score(self.label, self.pred)

    def balanced_accuracy(self):
        return balanced_accuracy_score(self.label, self.pred)

    def log_loss(self):
        return log_loss(self.label, self.score)
        
    # Methods for getting each plot    
    def rejection(self):
        rejection_plot(self.name, self.label, self.score)
        plt.show()
        
    def inverse_roc(self):
        inverse_roc_plot(self.name, self.label, self.score)
        plt.show()
    
    def significance(self):
        significance_plot(self.name, self.label, self.score)
        plt.show()
        
    def precision_recall(self):
        precision_recall_plot(self.name, self.label, self.score)
        plt.show()

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