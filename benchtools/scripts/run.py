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
import argparse
import pickle
import os.path 
import pandas as pd
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
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

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

# Importing the metrics
from sklearn.metrics import precision_score, log_loss, recall_score, f1_score, balanced_accuracy_score

# Benchtools
from benchtools.src.clustering import build_features
from benchtools.src.datatools import separate_data
from benchtools.src.metrictools import optimal_threshold, rejection_plot, inverse_roc_plot, significance_plot, \
     precision_recall_plot, compare_metrics, compare_metrics_plot

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

def TensorflowClassifier(input_shape):

    # Creating the model
    # Here are the layers with batch normalization, the drop out rate and the activations
    '''
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
    '''
    model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu'),   
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu'),   
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu'),   
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

def training(X_train, X_test, y_train, y_test, classifiers, path, models_name, dimension_reduction=None):
    
    models = []

    for scaler, clf in tqdm(classifiers):
        
        name = clf.__class__.__name__
        
        # For tensorflow the training is different
        if name == 'Sequential' :
            # Scaling the data
            X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
            X_test[X_test.columns] = scaler.fit_transform(X_test[X_test.columns])
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
            
            model.save(os.path.join(path,'tf_model_{}.h5'.format(models_name)))

        # For the sklearn algoritms
        else:
            # Simple pipeline
            if dimension_reduction is None:
                model = Pipeline(steps=[('ss', scaler), ('clf', clf)])
            else:
                model = Pipeline(steps=[('ss', scaler), ('dr', dimension_reduction), ('clf', clf)])

            # Training the model  
            model.fit(X_train, y_train) 
            
            # Saving into a list
            models.append((name,model))

    pickle.dump(models, open(os.path.join(path,'sklearn_models_{}.sav'.format(models_name)), 'wb'))
    print('Models saved') 

def evaluate(X_test, y_test, models):
   # To save the output
    clfs = []
    
    for name, model in tqdm(models):

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
        
        # For tensorflow the prediction is different
        else:
            y_score = model.predict(X_test)
            # Getting the threshold to make class predictions (0 or 1)
            threshold = optimal_threshold(y_test, y_score)
            y_pred = (model.predict(X_test) > threshold).astype("int32")
            clfs.append(classifier(name, y_score, y_pred, y_test))
    
    return clfs


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
    parser.add_argument('--nevents', type=int, default=100000, help='Number batches [default: 100,000]. If all_data is True, then this flag has no effect')
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
            chunksize = ceil(1100000/N_BATCH)
        else:
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
            chunksize = ceil(1000000/N_BATCH) 
        else:
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
        # Splitting in training and testis sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    else:  
        # Separating characteristics from label
        X_test, y_test = separate_data(df, standarize=False)
        # Dropping the mass to make the classification model-fre
        X_test.drop(['m_j1', 'm_j2', 'm_jj'], axis=1, inplace=True)

    if TRAINING is True:
        # Scalers and classifiers
        classifiers = [(MinMaxScaler(feature_range=(-1,1)), TensorflowClassifier(input_shape = [X_train.shape[1]])),
                        (StandardScaler(), RandomForestClassifier(random_state=1)),
                        (RobustScaler(), GradientBoostingClassifier(random_state=4)),
                        (RobustScaler(), QuadraticDiscriminantAnalysis()), 
                        (StandardScaler(), MLPClassifier(random_state=7)),
                        (StandardScaler(), KMeans(n_clusters=2, random_state=15))
                        ]


        print('TRAINING ALGORITHMS')

        training(X_train, X_test, y_train, y_test, classifiers, PATH_RAW, NAME_MODELS)

    print('GETTING PREDICTIONS AND SCORES')

    # Sklearn algorithms
    models = pickle.load(open(os.path.join(PATH_RAW,'sklearn_models_{}.sav'.format(NAME_MODELS)), 'rb'))

    # Tensorflow algorithm
    tf_model = load_model(os.path.join(PATH_RAW,'tf_model_{}.h5'.format(NAME_MODELS)))
    models.append(('TensorflowClassifier', tf_model))

    # Evaluation
    clfs = evaluate(X_test, y_test, models)

    # Getting the values to plot

    names = [clf.name for clf in clfs]
    scores = [clf.score for clf in clfs]
    preds = [clf.pred for clf in clfs]      
    labels = [clf.label.to_numpy() for clf in clfs] 

    # Adding algorithms trained and evaluated externaly
    if PATH_EXT_CLF != None:
        print('LOADING DATA FROM EXTERNAL ALGORITHMS')

        # Reading the list of files
        with open('{}'.format(PATH_EXT_CLF)) as f:
            external_clfs = [line.rstrip('\n') for line in f]
        
        # Adding the information to the existing lists
        for file in external_clfs:
            clf = pickle.load(open(os.path.join(PATH_RAW,file), 'rb'))
            names.append(clf.name)
            scores.append(clf.scores)
            preds.append(clf.pred)
            labels.append(clf.label)

    print('COMPARING METRICS')

    print('Classifiers to compare:')
    for name in names:
        print(name)


    # Plotting metrics

    rejection_plot(names=names, labels=labels, probs=scores)
    plt.savefig(os.path.join(PATH_OUT,'rejection.png'), bbox_inches='tight')
    plt.clf()

    inverse_roc_plot(names=names, labels=labels, probs=scores)
    plt.savefig(os.path.join(PATH_OUT,'inverse_roc.png'), bbox_inches='tight')
    plt.clf()

    significance_plot(names=names, labels=labels, probs=scores)
    plt.savefig(os.path.join(PATH_OUT,'significance.png'), bbox_inches='tight')
    plt.clf()

    precision_recall_plot(names=names, labels=labels, probs=scores)
    plt.savefig(os.path.join(PATH_OUT,'precision-recall.png'), bbox_inches='tight')
    plt.clf()

    # Metrics
    log = compare_metrics(names, scores, preds, labels)

    # Printing values to text
    with open(os.path.join(PATH_OUT,'metrics.txt'), "a") as f:
        print(tabulate(log, headers='keys', tablefmt='psql'), file=f)

    # Getting the name of the metrics
    metrics = log.columns.tolist()

    # Plotting the metrics
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for metric,color in zip(metrics,color_list):
        compare_metrics_plot(log, metric, color=color)
        plt.savefig(os.path.join(PATH_OUT,'{}_barh.png'.format(metric)), bbox_inches='tight')
        plt.clf()

if __name__ == "__main__":
    main()