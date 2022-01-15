import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, precision_score, log_loss, recall_score, classification_report, f1_score, average_precision_score

def roc_curve_and_score(label, pred_proba):
    """Returns the false positive rate (fpr), true positive rate(tpr) and
    the area under the curve (auc) of the ROC curve.

    Parameters
    ----------
    label : serie
        True binary labels.

    pred_proba : serie
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or non-thresholded measure of decisions.

    Returns
    ------
    fpr: 
        ndarray of shape (>2,)

    tpr:
        ndarray of shape (>2,)
    auc:
        float
    """
    fpr, tpr, _ = roc_curve(label, pred_proba.ravel())
    roc_auc = roc_auc_score(label, pred_proba.ravel())
    return fpr, tpr, roc_auc

def pr_curve_and_score(label, pred_proba):
    """Returns the precision, recall and the average precision (ap) 
    score for the precision-recall curve (PRc)

    Parameters
    ----------
    label : serie
        True binary labels.

    pred_proba : serie
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or non-thresholded measure of decisions.

    Returns
    ------
    fpr: 
        ndarray of shape (>2,)

    tpr:
        ndarray of shape (>2,)
    auc:
        float
    """
    precision, recall, _ = precision_recall_curve(label, pred_proba.ravel())
    ap_score = average_precision_score(label, pred_proba.ravel())
    return precision, recall, ap_score

def performance_metrics(name, label, pred_label, pred_prob=None):
    """Calculates the recall, precision, f1 score and logarithmic loss.
    Prints a classification report.

    Parameters
    ----------
     name: string
        Name of the classificator.

    label : serie
        True label of every event.

    pred_label: serie
        Predicted label for every event.
    
    pred_prob: serie
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or non-thresholded measure of decisions. Default is None

    Returns
    ------
    DataFrame
        DataFrame with the name, recall, precision, f1 score.
        Logarithmic loss if pred_prob was passed.
    """

    print("="*30)
    print(name)

    print('****Results****')

    # Calculating metrics
    precision = precision_score(label, pred_label)
    f1 = f1_score(label, pred_label)
    recall = recall_score(label, pred_label)
    if pred_prob: 
        ll = log_loss(label, pred_prob)
        # Naming the columns
        log_cols=["Classifier", "Recall", "Precision", "F1 score", "Log Loss"]
        # Inserting the data in the dataframe
        log_entry = pd.DataFrame([[name, recall, precision*100, f1, ll]], columns=log_cols)
    else:
        log_cols=["Classifier", "Recall", "Precision", "F1 score"]
        # Inserting the data in the dataframe
        log_entry = pd.DataFrame([[name, recall, precision*100, f1]], columns=log_cols)

    # Print the report
    print(classification_report(label, pred_label, target_names=['background','signal']))

    return log_entry

def sig_eff_bkg_rej(names, label, probs, colors):
    """Plots the signal efficiency (tpr) vs. the background rejection (1-fpr).

    Parameters
    ----------
    names : list
        Name of the algorithms.

    label: serie
        True label of every event.

    probs : list
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or non-thresholded measure of decisions.

    colors: list
        List of colors for the curves.

    Returns
    ------
    ax:
        The axis for the plot.
    """
    # Creating the figure an the axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    # Setting some parameters
    matplotlib.rcParams.update({'font.size': 14})
    plt.grid()

    # Plotting the curves
    for name, prob, color in zip(names, probs, colors):
        fpr, tpr, roc_auc = roc_curve_and_score(label, prob)
        plt.plot(1-fpr, tpr, color=color, lw=2,
                label='{} AUC={:.3f}'.format(name, roc_auc))

    # Plotting the line for a random classificator
    plt.plot([1, 0], [0, 1], color='navy', lw=1, linestyle='--')
    # Adding the information to the plot
    plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Background rejection')
    plt.ylabel('Signal efficiency')
    plt.title('Signal efficiency vs. Background rejection')

    return ax

def sig_eff_inv_bkg_eff(names, label, probs, colors):
    """Plots the signal efficiency (tpr) vs. the inverse of 
    background efficiency (1-fpr).

    Parameters
    ----------
    names : list
        Name of the algorithms.

    label: serie
        True label of every event.

    probs : list
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or non-thresholded measure of decisions.

    colors: list
        List of colors for the curves.
        
    Returns
    ------
    ax:
        The axis for the plot.
    """

    # Creating the figure an the axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    # Setting some parameters
    matplotlib.rcParams.update({'font.size': 14})
    plt.grid()
    
    # Plotting the curves
    for name, prob, color in zip(names, probs, colors):
        fpr, tpr, roc_auc = roc_curve_and_score(label, prob)
        plt.plot(tpr, 1/fpr, color=color, lw=2,
            label='{} ROC'.format(name))

    # Adding the information to the plot
    plt.legend(loc="upper right")
    plt.xlabel('$\epsilon_{sig}$')
    plt.ylabel('$1/\epsilon_{bkg}$')
    plt.title('ROC')
    
    return ax

def precision_recall_plot(names, label, probs, colors):
    """Plots precision vs. recall for different decision tresholes.

    Parameters
    ----------
    names : list
        Name of the algorithms.

    label: serie
        True label of every event.

    probs : list
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or non-thresholded measure of decisions.

    colors: list
        List of colors for the curves.

    Returns
    ------
    ax:
        The axis for the plot.
    """
    # Creating the figure an the axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    # Setting some parameters
    matplotlib.rcParams.update({'font.size': 14})
    plt.grid()

    # Plotting the curves
    for name, prob, color in zip(names, probs, colors):
        precision, recall, ap_score = pr_curve_and_score(label, prob)
        plt.plot(recall, precision, color=color, lw=2,
         label='{} AP={:.3f}'.format(name,ap_score))

    # Plotting the line for a random classificator
    plt.plot([0, 0], [0, 0], color='navy', lw=1, linestyle='--')
    # Adding the information to the plot
    plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    return ax
