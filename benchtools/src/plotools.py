import os
from tokenize import Triple
from PIL import Image
import matplotlib.pyplot as plt

def bkg_sig_hist(df, variable, label, xlabel=None, ylabel='Events density', n_bins=50):
    '''Plot two distributions on the same figure given a
    DataFrame with a column for the variable to plot and a label.

    Parameters
    ----------
    df : DataFrame
        Data for the events

    variable  : str
        Name of the variable to plot

    label : str
        Name of the column with an integer label
    
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
    '''

    # Getting the labels
    labels = df[label].unique()
    colors = ['b','r','y']
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    for integer, color in zip(labels, colors):
    # Plotting the histogramas
        df_plot = df.loc[df.loc[:,label]==integer]
        df_plot = df_plot[variable]
        if len(labels)==2:
            if integer == 0:
                df_plot.plot.hist(bins=n_bins, facecolor=color, alpha=0.2, label='background'.format(integer), density=True)
            else:
                df_plot.plot.hist(bins=n_bins, facecolor=color, alpha=0.2, label='signal'.format(integer), density=True)
        else:
            df_plot.plot.hist(bins=n_bins, facecolor=color, alpha=0.2, label='label {}'.format(integer), density=True)
    
    # Adding information to the plot
    if xlabel != None:
        plt.xlabel(xlabel)
    else: 
        plt.xlabel(variable)
        
    plt.ylabel(ylabel)
    plt.legend(loc='upper right')
    plt.title('Distribution of {}'.format(variable))
    
    return ax

def bkg_sig_scatter(df, x, y, label='label', title=None):
    '''Plot two scatter plots on the same figure on different axis, 
    for signal and background.

    Parameters
    ----------
    df : DataFrame
        Data for the events

    x  : str
        Name of the variable to plot on the x-axis

    y : str
        Name of the variable to plot on the y-axis
    
    label : str
        Name of the column with the boolean information for signal or background (default is label)
    
    title : str
        Title for the plot (default is None)

    Returns
    ------
    ax1, ax2 : 
        The axis of the plots
    '''
    fig = plt.figure(figsize=[12, 4])

    # Creating the subplots
    ax1 = fig.add_subplot(1,2, 1)
    ax2 = fig.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)

    # Signal
    df_sig = df[df[label]==1]
    xsig = df_sig[x]
    ysig= df_sig[y]
    signal= ax2.scatter(xsig, ysig, c='r', alpha=0.5)
    ax2.label_outer()

    # Background
    df_bkg = df[df[label]==0]
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
    '''Plots the distribution of the test and the prediction on an axis on the same plot, 
    separating signal and background.

    Parameters
    ----------
    df : DataFrame
        DataFrame with the features for each event

    variable  : Pandas Serie
        Variable from the DataFrame to plot

    n_bins : int
        Number of bins (default is 50)

    log : bool
        Logarithmic scale for the plot (default is False)

    Returns
    ------
    ax : 
        The axis of the plot
    '''
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

def create_png(namedf, df, variables, keyname, path, nbins=50, type='distribution', title=False):
    '''Creates multiple .png images using bkg_sig_hist or bkg_sig_scatter.

    Parameters
    ----------
    namedf : str
        Name for the dataframe
    
    df  : DataFrame
        DataFrame with the features for each event

    variables : list
        List of features or list of tuples

    keyname : str
        Key for all the images names

    path : str
        Path to save the images

    nbins : int
        Number of bins for distribution plot (default is 50)

    type : str
        'distribution' to use bkg_sig_hist or 'scatter' to use bkg_sig_scatter (default is 'distribution')
    
    title : bool
        True for using the titles in spanish (default is False)

    Returns
    ------
    list_images : list 
        List for the path of the images
    '''
    list_images= []
    for variable in variables:
        # Plotting
        fig = plt.figure(facecolor='white')
        if type == 'distribution':
            bkg_sig_hist(df, variable=variable, label='label', n_bins=nbins)
            if title is True:
                # Title (in spanish but can be changed)
                plt.title('{}: distribución de {}'.format(namedf, variable))
            # Defining path and name of the files
            filename = os.path.join(path,'{}-{}-{}.png'.format(keyname,namedf,variable))
        if type == 'scatter':
            if title is True:
                bkg_sig_scatter(df, variable[0], variable[1], title='{}: correlación entre {} y {}'.format(namedf,variable[0], variable[1]))
            else:
                bkg_sig_scatter(df, variable[0], variable[1])
            # Defining path and name of the files
            filename = os.path.join(path,'{}-{}-{}v{}.png'.format(keyname,namedf,variable[0], variable[1]))
        # Saving the path of each file
        list_images.append(filename)
        # Saving the figure as a png
        plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor(),edgecolor='none')
        plt.close()
    return list_images

def image_grid(rows, columns, images, name, path, remove=False):
    '''Grid row x columns of png images as a png file.

    Parameters
    ----------
    rows : int
        Number of rows

    columns : int
        Number of columns

    images : list
        List with the paths of the images

    name : str
        Name to save the png

    path : str
        Path to save the image

    remove : bool
        True if the list of images should be removed after creating the grid (default is False)

    Returns
    ------
    None
    '''
    # Getting sizes of the images
    width = []
    height = []
    for image in images:
        img = Image.open(image)
        w, h = img.size
        width.append(w)
        height.append(h)

    # And max dimensions
    wmax = max(width)
    hmax = max(height)

    new_image = Image.new('RGB', (columns*wmax, rows*hmax), color = 'white')
    
    # If the number of images is odd create a white one and add it to the list
    if len(images) % 2 != 0:
        img = Image.new("RGB", (wmax, hmax), (255, 255, 255))

        img.save(os.path.join(path,'whiteimage.png'), "PNG")
        images.append(os.path.join(path,'whiteimage.png'))
    
    # Create the grid
    ii = 0
    for row in range(rows):
        for column in range(columns):
            img = Image.open(images[ii])
            new_image.paste(img, (column*wmax, row*hmax))
            ii+=1

    # Save it
    new_image.save(os.path.join(path,'{}.png'.format(name)))

    # Deleting the images from the list
    if remove is True:
        for img in images: 
            os.remove(img)
        
    return None