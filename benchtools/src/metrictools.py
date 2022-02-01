import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, precision_score, log_loss, recall_score, classification_report, f1_score, average_precision_score

LIST_COLORS = ['darkorange', 'green', 'crimson', 'blue', 'green'
    , 'red', 'purple', 'pink', 'gray', 'olive', 'cyan', 'indigo'
    ,'salmon','gold', 'aquamarine', 'bluevioles', 'magenta', 'darkred'
    ,'sandybrown', 'darkseagreen','deepskyblue', 'deeppink']

def roc_curve_and_score(label, pred_proba):
    '''Returns the false positive rate (fpr), true positive rate(tpr) and
    the area under the curve (auc) of the ROC curve.

    Parameters
    ----------
    label : ndarray
        True binary labels.

    pred_proba : ndarray
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
    '''
    fpr, tpr, _ = roc_curve(label.ravel(), pred_proba.ravel())
    roc_auc = roc_auc_score(label.ravel(), pred_proba.ravel())
    return fpr, tpr, roc_auc

def optimal_threshold(label, pred_proba):
    '''Returns optimal treshold maximizing the true positive rate count.

    Parameters
    ----------
    label : ndarray
        True binary labels.

    pred_proba : ndarray
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or non-thresholded measure of decisions.

    Returns
    ------
    treshold: 
        float
    '''
    fpr, tpr, thresholds = roc_curve(label.ravel(), pred_proba.ravel())
    # Maximize the function
    optimal_idx = np.argmax(tpr - fpr)
    # Get the treshold for the max value of the function
    threshold = thresholds[optimal_idx]
    return threshold

def roc_plot(names, labels, probs, colors=None):
    '''Plots the background efficiency (fpr) vs. signal efficiency (tpr)

    Parameters
    ----------
    names : list
        Name of the algorithms.

    labels: ndarray
        True label of every event.

    probs : list
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or non-thresholded measure of decisions.

    colors: string
        List of specific colors for the plots (default is None)

    Returns
    ------
    ax:
        The axis for the plot.
    '''

    if colors is None:
        # Selecting colors
        colors = LIST_COLORS[:len(names)]
        
    # Creating the figure an the axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    # Setting some parameters
    matplotlib.rcParams.update({'font.size': 14})
    plt.grid()

    # Plotting the curves
    # In case there are different labels
    if type(labels)==list:
        for name, label, prob, color in zip(names, labels, probs, colors):
            fpr, tpr, roc_auc = roc_curve_and_score(label, prob)
            plt.plot(fpr, tpr, color=color, lw=2,
                    label='{} AUC={:.3f}'.format(name, roc_auc))
    else:
        for name, prob, color in zip(names, probs, colors):
            fpr, tpr, roc_auc = roc_curve_and_score(labels, prob)
            plt.plot(fpr, tpr, color=color, lw=2,
                    label='{} AUC={:.3f}'.format(name, roc_auc))

    # Plotting the line for a random classifier
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random classification')

    # Adding information to the plot
    plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Background efficiency')
    plt.ylabel('Signal efficiency')
    plt.title('ROC curve')
    plt.show()

    return ax

def rejection_plot(names, labels, probs, colors=None):
    '''Plots the signal efficiency (tpr) vs. the background rejection (1-fpr).

    Parameters
    ----------
    names : list
        Name of the algorithms.

    label: ndarray
        True label of every event.

    probs : list
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or non-thresholded measure of decisions.

    colors: list
        List of specific colors for the curves (default is None)

    Returns
    ------
    ax:
        The axis for the plot.
    '''
    # Selecting colors in case it weren't specified
    if colors is None:
        colors = LIST_COLORS[:len(names)]

    # Creating the figure an the axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Setting some parameters
    matplotlib.rcParams.update({'font.size': 14})
    plt.grid()

    # Plotting the curves
    # Different labels
    if type(labels)==list:
        for name, label, prob, color in zip(names, labels, probs, colors):
            fpr, tpr, roc_auc = roc_curve_and_score(label, prob)
            plt.plot(1-fpr, tpr, color=color, lw=2,
                    label='{} AUC={:.3f}'.format(name, roc_auc))

    # Same label
    else:
        for name, prob, color in zip(names, probs, colors):
            fpr, tpr, roc_auc = roc_curve_and_score(labels, prob)
            plt.plot(1-fpr, tpr, color=color, lw=2,
                    label='{} AUC={:.3f}'.format(name, roc_auc))

    # Plotting the line for a random classifier
    plt.plot([1, 0], [0, 1], color='navy', lw=1, linestyle='--', label='Random classification')

    # Adding information to the plot
    plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Background rejection')
    plt.ylabel('Signal efficiency')
    plt.title('Signal efficiency vs. Background rejection')

    return ax

def inverse_roc_plot(names, labels, probs, colors=None):
    '''Plots the signal efficiency (tpr) vs. rejection (1/fpr).

    Parameters
    ----------
    names : list
        Name of the algorithms.

    labels: ndarray
        True label of every event.

    probs : list
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or non-thresholded measure of decisions.

    colors: list
        List of specific colors for the curves (default is None)
        
    Returns
    ------
    ax:
        The axis for the plot.
    '''
    # To ignore division by zero error
    np.seterr(divide='ignore')

    # Selecting colors in case they weren't specified
    if colors is None:
        colors = LIST_COLORS[:len(names)]

    # Creating the figure an the axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Setting some parameters
    matplotlib.rcParams.update({'font.size': 14})
    plt.grid()
    
    # Plotting the curves
    # Different labels
    if type(labels)==list:
        for name, label, prob, color in zip(names, labels, probs, colors):
            fpr, tpr, _ = roc_curve_and_score(label, prob)
            plt.plot(tpr, 1/fpr, color=color, lw=2,
                    label='{}'.format(name))

    # Same labels
    else:
        for name, prob, color in zip(names, probs, colors):
            fpr, tpr, _ = roc_curve_and_score(labels, prob)
            plt.plot(tpr, 1/fpr, color=color, lw=2,
                    label='{}'.format(name))

    # Adding information to the plot
    plt.legend(loc="upper right")
    plt.xlabel('Signal efficiency')
    plt.ylabel('Rejection')
    plt.title('ROC')
    
    return ax

def significance_plot(names, labels, probs, colors=None):
    '''Plots the signal efficiency (tpr) vs. the significance
    improvement (tpr/sqrt(fpr)).

    Parameters
    ----------
    names : list
        Name of the algorithms.

    labels: ndarray
        True label of every event.

    probs : list
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or non-thresholded measure of decisions.

    colors: list
        List of specific colors for the curves (default is None)
        
    Returns
    ------
    ax:
        The axis for the plot.
    '''
    # To ignore division by zero or NaN error
    np.seterr(divide='ignore', invalid='ignore')

    # Selecting colors in case they weren't specified
    if colors is None:
        colors = LIST_COLORS[:len(names)]

    # Creating the figure an the axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    # Setting some parameters
    matplotlib.rcParams.update({'font.size': 14})
    plt.grid()
    
    # Plotting the curves
    # For different labels
    if type(labels)==list:
            for name, label, prob, color in zip(names, labels, probs, colors):
                fpr, tpr, _ = roc_curve_and_score(label, prob)
                plt.plot(tpr, tpr/np.sqrt(fpr), color=color, lw=2,
                        label='{}'.format(name))

    # Same labels
    else:
        for name, prob, color in zip(names, probs, colors):
            fpr, tpr, _ = roc_curve_and_score(labels, prob)
            plt.plot(tpr, tpr/np.sqrt(fpr), color=color, lw=2,
                    label='{}'.format(name))

    # Adding information to the plot
    plt.legend(loc="upper right")
    plt.xlabel('Signal efficiency')
    plt.ylabel('Significance improvement')
    plt.title('ROC')
    
    return ax

def pr_curve_and_score(label, pred_prob):
    '''Returns the precision, recall and the average precision (ap) 
    score for the precision-recall curve (PRc)

    Parameters
    ----------
    label : ndarray
        True binary labels.

    pred_proba : ndarray
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
    '''
    precision, recall, _ = precision_recall_curve(label.ravel(), pred_prob.ravel())
    ap_score = average_precision_score(label.ravel(), pred_prob.ravel())

    return precision, recall, ap_score

def precision_recall_plot(names, labels, probs, colors=None):
    '''Plots precision vs. recall for different decision tresholes.

    Parameters
    ----------
    names : list
        Name of the algorithms.

    labels: ndarray
        True label of every event.

    probs : list
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or non-thresholded measure of decisions.

    colors: list
        List of specific colors for the curves (default is None)

    Returns
    ------
    ax:
        The axis for the plot.
    '''
    # Selecting colors in case they weren't specified
    if colors is None:
        colors = LIST_COLORS[:len(names)]

    # Creating the figure an the axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Setting some parameters
    matplotlib.rcParams.update({'font.size': 14})
    plt.grid()

    # Plotting the curves
        # For different labels
    if type(labels)==list:
            for name, label, prob, color in zip(names, labels, probs, colors):
                precision, recall, ap_score = pr_curve_and_score(label, prob)
                plt.plot(recall, precision, color=color, lw=2,
                        label='{} AP={:.3f}'.format(name,ap_score))

    # Same labels
    else:
        for name, prob, color in zip(names, probs, colors):
            precision, recall, ap_score = pr_curve_and_score(labels, prob)
            plt.plot(recall, precision, color=color, lw=2,
                    label='{} AP={:.3f}'.format(name,ap_score))

        _, counts = np.unique(labels, return_counts=True)
        ratio = counts[1]/counts[0]
        print(ratio)
        # Plotting the line for a random classifier
        plt.axhline(y=ratio, color='navy', lw=1, linestyle='--', label='Random classification')
    
    # Adding information to the plot
    plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    return ax

def performance_metrics(name, label, pred_label, pred_prob=None):
    '''Calculates the recall, precision, f1 score and logarithmic loss.
    Prints a classification report.

    Parameters
    ----------
     name: string
        Name of the classificator.

    label : ndarray
        True label of every event.

    pred_label: ndarray
        Predicted label for every event.
    
    pred_prob: ndarray
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or non-thresholded measure of decisions. Default is None

    Returns
    ------
    DataFrame
        DataFrame with the name, recall, precision, f1 score.
        Logarithmic loss if pred_prob was passed.
    '''

    print("="*30)
    print(name)

    print('****Results****')

    # Calculating metrics
    precision = precision_score(label, pred_label)
    f1 = f1_score(label, pred_label)
    recall = recall_score(label, pred_label)
    if pred_prob is not None: 
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