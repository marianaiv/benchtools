'''
Script to convert the results of GAN-AE into the format needed for the benchmarking with run.py
'''
import os
import h5py
import pickle
import argparse
import numpy as np
import pandas as pd
#from scripts.run import classifier
from benchtools.src.metrictools import optimal_threshold

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

DATA = flags.path 
NBOX = flags.box
RD = flags.RD

if RD:
    sample = 'RnD_distances.h5'
    # Reading the distances
    dist_bkg = pd.DataFrame(np.array(h5py.File(os.path.join(DATA,sample), 'r')['bkg']), columns=['y_score'])
    dist_sig = pd.DataFrame(np.array(h5py.File(os.path.join(DATA,sample), 'r')['sig1']), columns=['y_score'])
else:
    sample = 'BB{}_distances.h5'.format(NBOX)
    dist_bkg = pd.DataFrame(np.array(h5py.File(os.path.join(DATA,sample), 'r')['bkg']), columns=['y_score'])
    dist_sig = pd.DataFrame(np.array(h5py.File(os.path.join(DATA,sample), 'r')['sig']), columns=['y_score'])



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