import matplotlib.pyplot as plt         # Plots

def bkg_sig_hist(df, variable, label, xlabel=None, ylabel='Events density', n_bins=50):
    """Plot two distributions on the same figure and axis given a
    DataFrame with a column for the variable to plot and the label:
    signal or background (or true and false).

    Parameters
    ----------
    df : DataFrame
        Data for the events

    variable  : str
        Name of the variable to plot

    label : str
        Name of the column with the boolean information for signal or background
    
    xlabel : str
        Label for the x-axis (default is None)
    
    ylabel : str
        Label for the y-axis (default is Events density)

    n_bins : int
        Number of bins for the plots

    Returns
    ------
    ax : 
        The axis for the plots
    """
    # Getting the data for signal or background
    sig = df.loc[df.loc[:,label]==1]
    bkg = df.loc[df.loc[:,label]==0]
    
    # And specifically for what we want to plot
    sig = sig[variable]
    bkg = bkg[variable]
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    # Plotting the histogramas
    bkg.plot.hist(bins=n_bins, facecolor='b', alpha=0.2, label='background', density=True)
    sig.plot.hist(bins=n_bins, facecolor='r', alpha=0.2, label='signal', density=True)
    
    # Adding information to the plot
    if xlabel != None:
        plt.xlabel(xlabel)
    else: 
        plt.xlabel(variable)
        
    plt.ylabel(ylabel)
    plt.legend(loc='upper right')
    plt.title('Distribution of {}'.format(variable))
    
    return ax

def bkg_sig_scatter(df, x, y, title=None):
    """Plot two scatter plots on the same figure on different axis, 
    for signal and one  background.

    Parameters
    ----------
    xbkg : Pandas Serie
        Data for one of the distributions

    ybkg  : Pandas Serie
        Data for one of the distributions

    xsig : Pandas Serie
        Data for one of the distributions

    xbkg  : Pandas Serie
        Data for one of the distributions
    
    xlabel : str
        Label for the x-axis (default is None)
    
    ylabel : str
        Label for the y-axis (default is None)
    
    title : str
        Title for the plot (default is None)

    Returns
    ------
    ax1, ax2 : 
        The axis of the plots
    """
    fig = plt.figure(figsize=[12, 4])

    # Creating the subplots
    ax1 = fig.add_subplot(1,2, 1)
    ax2 = fig.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)

    # Signal
    df_sig = df[df['label']==1]
    xsig = df_sig[x]
    ysig= df_sig[y]
    signal= ax2.scatter(xsig, ysig, c='r', alpha=0.5)
    ax2.label_outer()

    # Background
    df_bkg = df[df['label']==0]
    xbkg = df_bkg[x]
    ybkg= df_bkg[y]
    bkg = ax1.scatter(xbkg, ybkg, c='b', alpha=0.5)
    ax1.set(ylabel=y)
    #ax1.set_title('bkg')

    # Adding information to the plot
    ax2.legend((signal, bkg), ('Signal', 'Background'), loc='upper right', prop={'size': 12})
    fig.suptitle(title)
    fig.text(0.5, 0.04, x, ha='center', va='center')
    plt.subplots_adjust(wspace=0);
    
    return ax1, ax2

def pred_test_hist(df, variable, ypred='y_pred', ytest='y_test', n_bins=50, log=False):
    """Plots the distribution of the test and the prediction on an axis on the same plot, 
    separating signal and background.

    Parameters
    ----------
    df : DataFrame
        DataFrame with the features for each event

    variable  : Pandas Serie
        Variable from the DataFrame to plot

    n_bins : Pandas Serie
        Number of bins (default is 50)

    log : bool
        Logarithmic scale for the plot (default is False)

    Returns
    ------
    ax : 
        The axis of the plot
    """
    # Separating signal from background for both the test and pred
    pred_bkg = df[df[ypred] == 0]
    pred_sig = df[df[ypred] == 1]

    bkg_test = df[df[ytest] == 0]
    sig_test = df[df[ytest] == 1]
    
    # Obtaining the variable to plot
    pred_bkg = pred_bkg[variable]
    pred_sig = pred_sig[variable]

    bkg_test = bkg_test[variable]
    sig_test = sig_test[variable]
    
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    title = 'Distribution of {}'.format(variable)
    
    # Plotting the histograms for the pred as a line
    pred_bkg.plot.hist(bins=n_bins, histtype='step', log=log, color='green', label='background pred', density=True)
    pred_sig.plot.hist(bins=n_bins, histtype='step', log=log, color='orange', label='signal pred', density=True)
    
    # And the histograms for the test completely colored
    bkg_test.plot.hist(bins=n_bins, log=log, facecolor='blue', alpha=0.2, label='background', density=True)
    sig_test.plot.hist(bins=n_bins, log=log, facecolor='red', alpha=0.2, label='signal', density=True)

    # Adding information to the plot
    plt.xlabel(variable)
    plt.ylabel('Events density')
    plt.legend(loc='upper right')
    plt.title(title)
    
    return ax