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

### Libraries ### 
from importlib.resources import path
import os
import pickle
import argparse
import os.path 
import pandas as pd
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from math import ceil
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Importing the classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

# Benchtools
from benchtools.src.clustering import build_features
from benchtools.src.datatools import separate_data
from benchtools.src.metrictools import classifier, performance_metrics


def training(X_train, y_train, classifier, name_models, dimension_reduction=None):
    """Trains multiple sklearn binary classification algorithms and a tensorflow sequential model.

    Parameters
    ----------
    X_train : DataFrame
        Features for training
    
    X_test : DataFrame
        Features for testing

    y_train : Series
        True label of the train features

    y_test: Series
        True label of the test features

    classifiers: list
        List of tuples (data scaler, classifier)

    path: str
        Path to save the models

    models_name: str
        Name to add to the saved files

    dimension_reduction : function
        Function to use for reducing dimensions. Default is None

    Returns
    ------
    File 
        h5 and joblib files. Saved trained models
    """

    #for scaler, clf in tqdm(classifiers):
    scaler = classifier[0]
    clf = classifier[1]

    name = clf.__class__.__name__
    
    # Simple pipeline
    if dimension_reduction is None:
        model = Pipeline(steps=[('ss', scaler), ('clf', clf)])
    else:
        model = Pipeline(steps=[('ss', scaler), ('dr', dimension_reduction), ('clf', clf)])

    # Training the model  
    model.fit(X_train, y_train) 

    model_tuple = (name,model)
        
    # Saving into a pickle file
    with open("sklearn_models_{}.pckl".format(name_models), "wb") as f:
        pickle.dump(model_tuple, f)
    print('Model saved') 

def evaluate(X_test, y_test, model):
    """Get predictions and scores for multiple sklearn binary classification 
    algorithms and a tensorflow sequential model.

    Parameters
    ----------

    X_test : DataFrame
        Features for testing

    y_test: Series
        True label of the test features

    models : list
        List of tuples (name, trained classifier)

    train : bool
        If the model was trained in the current run. Defaul is False

    Returns
    ------
    clfs :  list
        List of classifier objects
    """
    
    
    #for name, model in tqdm(models):

    # Getting the prediction
    y_pred = model[1].predict(X_test)
    
    # Probability or distances
    y_score = model[1].predict_proba(X_test)

    one_classifier = classifier(model[0], y_score[:,1], y_pred, y_test)
    
    return one_classifier


def main():   
    tf.random.set_seed(125)

    # DEFAULT SETTINGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='../../data/', help='Folder containing the input files [Default: ../../data]')
    parser.add_argument('--out', type=str, default='../../logs/', help='Folder to save output files [Default: ../../logs]')
    parser.add_argument('--ext_clf', type=str, default=None, help='Path for the txt with the list of external classifiers to compare. The files in the list have to be in --dir if added [Default: None]')
    parser.add_argument('--nbatch', type=int, default=10, help='Number batches [default: 10]')
    parser.add_argument('--name', type=str, default='log', help='Name of the output folder. The folder is created in --out [Default: log]')
    parser.add_argument('--models', type=str, default='log', help='Name to save the models [Default: log]')
    parser.add_argument('--box', type=int, default=1, help='Black Box number, ignored if RD dataset [default: 1]')
    parser.add_argument('--RD',  default=False, action='store_true' ,help='Use RD data set [default: False')
    parser.add_argument('--nevents', type=int, default=100000, help='Number of events to use [default: 100,000]. If all_data is True, then this flag has no effect')
    parser.add_argument('--all_data', default=False, action='store_true', help='Use the complete dataset [default: False]')
    parser.add_argument('--training', default=False, action='store_true', help='To train the algorithms [default: False]')


    flags = parser.parse_args()

    PATH_RAW = flags.dir
    PATH_LOG = flags.out
    PATH_EXT_CLF = flags.ext_clf
    OUT_NAME = flags.name
    NAME_MODELS = flags.models
    RD = flags.RD
    N_EVENTS = flags.nevents
    ALL_DATA = flags.all_data
    N_BATCH = flags.nbatch
    TRAINING = flags.training

    # Path for saving all files created in one run
    PATH_OUT = os.path.join(PATH_LOG,OUT_NAME)

    # If the out path does not exists, creates it
    if not os.path.exists(PATH_OUT):
        os.makedirs(PATH_OUT)
    
    
    print('BUILDING FEATURES')

    if RD:
        # Getting the size for each chunk
        if ALL_DATA:
            N_EVENTS = 1100000
   
        chunksize = ceil(N_EVENTS/N_BATCH)

        # Building the features
        sample = 'events_anomalydetection.h5'
        path_sample = os.path.join(PATH_RAW,sample)
        filename = 'features_RD_{}'.format(N_EVENTS)

        print('Building features from the R&D dataset')
        build_features(path_data=path_sample, nbatch=N_BATCH, outname=filename, 
                    path_label=None, outdir=PATH_RAW, chunksize=chunksize)

    else:
        # Getting the size for each chunk
        if ALL_DATA:
            N_EVENTS = 1000000 

        chunksize = ceil(N_EVENTS/N_BATCH)

        # Building the features
        sample = 'events_LHCO2020_BlackBox{}.h5'.format(flags.box)
        label = 'events_LHCO2020_BlackBox{}.masterkey'.format(flags.box)
        path_sample = os.path.join(PATH_RAW,sample)
        path_label= os.path.join(PATH_RAW,label)
        filename = 'features_BB{}_{}'.format(flags.box,N_EVENTS)

        print('Building features from the BB{} dataset'.format(flags.box))
        build_features(path_data=path_sample, nbatch=N_BATCH, outname=filename, 
                    path_label=path_label, outdir=PATH_RAW, chunksize=chunksize)

    
    print('PREPARING THE DATA')
    
    df = pd.read_csv(os.path.join(PATH_RAW, '{}.csv'.format(filename)))

    if TRAINING or RD:
        # Separating characteristics from label
        X, y = separate_data(df, standarize=False)
        # Dropping the mass to make the classification model-fre
        X.drop(['m_j1', 'm_j2', 'm_jj'], axis=1, inplace=True)
        # Splitting in training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    else:  
        # Separating characteristics from label
        X_test, y_test = separate_data(df, standarize=False)
        # Dropping the mass to make the classification model-free
        X_test.drop(['m_j1', 'm_j2', 'm_jj'], axis=1, inplace=True)

    if TRAINING:
        # Scalers and classifiers
        classifier = (StandardScaler(), RandomForestClassifier(random_state=1))
        
        print('TRAINING ALGORITHMS')
        
        training(X_train, y_train, classifier, NAME_MODELS)

    
    print('GETTING PREDICTIONS AND SCORES')
   
    # Sklearn algorithms
    models = []
    with open("sklearn_models_{}.pckl".format(NAME_MODELS), "rb") as f:
        while True:
            try:
                models.append(pickle.load(f))
            except EOFError:
                break

    # Tensorflow algorithm
    tf_model = load_model(os.path.join(PATH_RAW,'tf_model_{}.h5'.format(NAME_MODELS)))
    models.append(('TensorflowClassifier', tf_model))

    # Evaluation
    clf = evaluate(X_test, y_test, models,train=TRAINING)

    print('COMPARING METRICS')

    # Plotting metrics
    # Legend size
    
    plt.rc('legend',fontsize='x-small')

    clf.rejection()
    plt.savefig(os.path.join(PATH_OUT,'rejection.png'), bbox_inches='tight')
    plt.clf()

    clf.inverse_roc()
    plt.savefig(os.path.join(PATH_OUT,'inverse_roc.png'), bbox_inches='tight')
    plt.clf()

    clf.significance()
    plt.savefig(os.path.join(PATH_OUT,'significance.png'), bbox_inches='tight')
    plt.clf()

    clf.precision_recall()
    plt.savefig(os.path.join(PATH_OUT,'precision-recall.png'), bbox_inches='tight')
    plt.clf()

    # Numeric metrics
    performance_metrics(clf.name, clf.label, clf.pred, clf.score)

if __name__ == "__main__":
    main()