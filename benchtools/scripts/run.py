### Getting the data ###
'''
The data used in this project can be donwloaded from zenodo:

R&D dataset
https://zenodo.org/record/2629073#.XjOiE2PQhEa
    Direct link:
    https://zenodo.org/record/2629073/files/events_anomalydetection.h5?download=1

Black Box:
https://zenodo.org/record/4536624
    Direct links: 
    - BB1: https://zenodo.org/record/4536624/files/events_LHCO2020_BlackBox1.h5?download=1
    - Masterkey: https://zenodo.org/record/4536624/files/events_LHCO2020_BlackBox1.masterkey?download=1
'''

### Pre-processing ###
import os
import argparse
import os.path 
import pandas as pd
import numpy as np
from tqdm import tqdm
from benchtools.src.clustering import build_features
from benchtools.src.datatools import separate_data
from math import ceil
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

# Importing the classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

from benchtools.src.metrictools import optimal_threshold, rejection_plot, inverse_roc_plot, significance_plot, precision_recall_plot

class classifier:
    def __init__(self, name, score, pred, label):
        self.name = name
        self.score = score
        self.pred = pred       
        self.label = label
    #def numeric_metrics()
        
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

def TensorflowClassifier(input_shape):

    # Creating the model
    # Here are the layers with batch normalization, the drop out rate and the activations
    model = keras.Sequential([
        layers.BatchNormalization(input_shape=input_shape),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),   
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),   
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),   
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),   
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])

    # Choosing the optimizer
    # Binary crossentropy for binary classification
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],)
    
    return model

def training_pipeline(X_train, y_train, X_test, y_test, classifiers, dimension_reduction=None):
    
    models = []

    for scaler, clf in tqdm(classifiers):
        
        try: name = clf.__class__.__name__ 
        
        except: name = 'TensorflowClassifier'

        if name != 'TensorflowClassifier':
            
            # Simple pipeline
            if dimension_reduction is None:
                model = Pipeline(steps=[('ss', scaler), ('clf', clf)])
            else:
                model = Pipeline(steps=[('ss', scaler), ('dr', dimension_reduction), ('clf', clf)])

                # Training the model
                try: model.fit(X_train) # For KMeans which is unsupervised
                except: model.fit(X_train, y_train)
            
                # Saving into a list
                models.append((name,model))

        # For tensorflow the training is different
        else:
            # Scaling the data
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
            # Getting model
            model = clf
            model.summary()
            # We use early stop to prevent the model from overfitting
            early_stopping = keras.callbacks.EarlyStopping(
                patience=20,
                min_delta=0.001,
                restore_best_weights=True,
            )
            # Training the model
            model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=512,
            epochs=200,
            callbacks=[early_stopping])
  
            model.save('../data/models/tf_model.h5')


    pickle.dump(models, open('../data/models/sklearn_models.sav', 'wb'))
    print('Models saved') 

def evaluate_pipeline(X_test, y_test, models):
   # To save the output
    clfs = []
    
    for name, model in tqdm(models):
        
        print(name)
        '''
        # Simple pipeline
        if dimension_reduction is None:
            model = Pipeline(steps=[('ss', scaler), ('clf', clf)])
        else:
            model = Pipeline(steps=[('ss', scaler), ('dr', dimension_reduction), ('clf', clf)])
        '''
        if name != 'TensorflowClassifier':
            # Getting the prediction
            y_pred = model.predict(X_test)
            
            # Probability or distances
            try: 
                y_score = model.predict_proba(X_test)

                clfs.append(classifier(name, y_score[:,1], y_pred, y_test))
            except: 
                # KMeans doesn't have a probability, so here we get 
                # the distances to each cluster
                y_score = model.transform(X_test)
                # The score for KMeans is defined differently
                norm = np.linalg.norm(y_score[:,1])
                clfs.append(classifier(name, 1-y_score[:,1]/norm, y_pred, y_test))
        
        # For tensorflow the prediction is done differently
        else:
            y_scores = model.predict(X_test)

            # Getting the threshold to make class predictions (0 or 1)
            threshold = optimal_threshold(y_test, y_scores[:,1])
            y_pred = (model.predict(X_test) > threshold).astype("int32")
            clfs.append(classifier(name, y_score[:,1], y_pred, y_test))
    
    return clfs

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='../data/', help='Folder containing the input files [Default: ../data]')
parser.add_argument('--out', type=str, default='../data/output', help='Folder to save output files [Default: ../data/output]')
parser.add_argument('--nbatch', type=int, default=10, help='Number batches [default: 10]')
parser.add_argument('--name', type=str, default='output', help='Name of the output file')
parser.add_argument('--box', type=int, default=1, help='Black Box number, ignored if RD dataset [default: 1]')
parser.add_argument('--RD',  type=bool, default=True ,help='Use RD data set [default: False')
parser.add_argument('--nevents', type=int, default=100000, help='Number batches [default: 100,000]. If all_data is True, then this flag has no effect')
parser.add_argument('--all_data', type=bool, default=False, help='Use the complete dataset [default: False]')
parser.add_argument('--training', type=bool, default=True, help='To train the algorithms [default: True]')

flags = parser.parse_args()

PATH_RAW = flags.dir
PATH_OUT = flags.out
OUT_NAME = flags.name
RD = flags.RD
N_EVENTS = flags.nevents
ALL_DATA = flags.all_data
N_BATCH = flags.nbatch
TRAINING = flags.training

# If the path does not exists, creates it
if not os.path.exists(os.path.join('..',PATH_OUT)):
    os.makedir(os.path.join('..',PATH_OUT))
    
print('BUILDING FEATURES')

if RD:
    # Getting the size for each chunk
    if ALL_DATA:
        chunksize = ceil(1100000/N_BATCH)
    else:
        chunksize = ceil(N_EVENTS/N_BATCH)

    # Building the features
    sample = 'events_anomalydetection.h5'
    print('Building features from the R&D dataset')
    build_features(path_data=PATH_RAW, nbatch=N_BATCH, outname='features_{}'.format(OUT_NAME), 
                path_label=None, outdir=PATH_OUT, chunksize=chunksize)

else:
    # Getting the size for each chunk
    if ALL_DATA:
        chunksize = ceil(1000000/N_BATCH) 
    else:
        chunksize = ceil(N_EVENTS/N_BATCH)

    # Building the features
    sample = 'events_LHCO2020_BlackBox{}.h5'.format(flags.box)
    print('Building features from the BB{} dataset'.format(flags.box))
    build_features(path_data=PATH_RAW, nbatch=N_BATCH, outname='features_{}'.format(OUT_NAME), 
                path_label=None, outdir=PATH_OUT, chunksize=chunksize)


print('GETTING DATA READY FOR TRAINING')

file_name = 'features_{}'.format(OUT_NAME)
df = pd.read_csv("..\data\{}.csv".format(file_name))

# Separating characteristics from label
X, y = separate_data(df, standarize=False)
# Dropping the mass to make the classification model-fre
X.drop(['m_j1', 'm_j2', 'm_jj'], axis=1)
# Splitting in training and testis sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Scalers and classifiers
classifiers = [(StandardScaler(), RandomForestClassifier()),
                (RobustScaler(), GradientBoostingClassifier()),
                (RobustScaler(), QuadraticDiscriminantAnalysis()), 
                (StandardScaler(), MLPClassifier()),
                (StandardScaler(), KMeans(n_clusters=2, random_state=15)),
                (MinMaxScaler(feature_range=(-1,1)), TensorflowClassifier(input_shape = [X_train.shape[1]]))
                ]


print('TRAINING ALGORITHMS')

training_pipeline(X_train, X_test, y_train, y_test, classifiers)

print('GETTING PREDICTIONS AND SCORES')

# Sklearn algorithms
models = pickle.load(open('../data/models/sklearn_models.sav', 'rb'))

# Tensorflow algorithm
tf_model = load_model('../data/models/tf_model.h5')
models.append('TensorflowClassifier', tf_model)

clfs = evaluate_pipeline(X_test, y_test, models)

print('LOADING DATA FROM EXTERNAL ALGORITHMS (soon)')


print('COMPARING METRICS')

# Getting the values to plot

names = [clf.name for clf in clfs]
scores = [clf.score for clf in clfs]
preds = [clf.pred for clf in clfs]      
labels = [clf.label for clf in clfs] 

rej = rejection_plot(names=names, labels=labels, probs=scores)
plt.savefig('test.png', bbox_inches='tight')