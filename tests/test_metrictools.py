import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from benchtools.src.metrictools import optimal_threshold, performance_metrics, compare_metrics, classifier

def test_optimal_threshold():
    np.random.seed(26)
    # random arrays with probabilities and zeros and ones
    pred_proba = np.random.rand(1,100)
    label = np.hstack((np.ones(9), np.zeros(91)))
    np.random.shuffle(label)
    # calculating thresholds
    fpr, tpr, thresholds = roc_curve(label.ravel(), pred_proba.ravel())

    # testing for an exact value
    assert optimal_threshold(label, pred_proba) == 0.9528972259126289

    pred_proba_zero = np.zeros(100)
    pred_proba_one = np.ones(100)

    # testing for all probabilities zero or all ones
    assert optimal_threshold(label, pred_proba_zero) == 1.0
    assert optimal_threshold(label,pred_proba_one) == 2.0

def test_performance_metrics():
    # importing classifier
    clf = pd.read_csv('data/test_metrics.csv')
    # calculating numeric metrics
    df = performance_metrics('clf', clf['label'], clf['pred'], clf['score'])

    # asserting each value
    assert df.loc[0, 'Recall'] == 0.8430767356387191
    assert df.loc[0, 'Precision'] == 39.62095800668954
    assert df.loc[0, 'F1 score'] == 0.5390765235738051
    assert df.loc[0, 'Log Loss'] == 17.614306499517163

    # checking all is zero if the classifier just guess zeros
    df = performance_metrics('zeros', clf['label'], np.zeros(len(clf['label'])))
    # asserting each value
    assert df.loc[0, 'Recall'] == 0.0
    assert df.loc[0, 'Precision'] == 0.0
    assert df.loc[0, 'F1 score'] == 0.0

    # checking a perfect classification
    df = performance_metrics('perfect', clf['label'], clf['label'])
    # asserting each value
    assert df.loc[0, 'Recall'] == 1.0
    assert df.loc[0, 'Precision'] == 100.0
    assert df.loc[0, 'F1 score'] == 1.0

def test_compare_metrics():
    # importing classifier
    clf = pd.read_csv('data/test_metrics.csv')
    df = compare_metrics(['clf'], [clf['score']], [clf['pred']], [clf['label']])
    # asserting each value
    assert df.loc['clf','Balanced accuracy'] == 0.7949953678193595
    assert df.loc['clf','Precision'] == 0.3962095800668954
    assert df.loc['clf','F1 score'] == 0.5390765235738051
    assert df.loc['clf','Recall'] == 0.8430767356387191

    # checking if the classifier just guess zeros
    df = compare_metrics(['clf'], [clf['score']], [np.zeros(len(clf['label']))], [clf['label']])
    # asserting each value
    assert df.loc['clf','Balanced accuracy'] == 0.5
    assert df.loc['clf','Precision'] == 0.0
    assert df.loc['clf','F1 score'] == 0.0
    assert df.loc['clf','Recall'] == 0.0

    # checking a perfect classification
    df = compare_metrics(['clf'], [clf['score']], [clf['label']], [clf['label']])
    # asserting each value
    assert df.loc['clf','Balanced accuracy'] == 1.0
    assert df.loc['clf','Precision'] == 1.0
    assert df.loc['clf','F1 score'] == 1.0
    assert df.loc['clf','Recall'] == 1.0
