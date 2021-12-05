import pandas as pd
import numpy as np
from io import StringIO
import os.path                          
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None  # default='warn'

def read_multifiles(filename, nbatch, outdir = '../data'): 
    """Returns a single DataFrame from multiple csv files named: file_0, file_1, ...

    If the argument `outdir` is not passed in, the default value is used.

    Parameters
    ----------
    filename : str
        Principal name of the set of files

    nbatch : int
        Number of files

    outdir : str
        Path where the files are located  (default is '../data')

    Returns
    ------
    df : Dataframe
        A single DataFrame with data from all the files.
    """
    
    # Listing the names of the files to upload in a list 
    names = ["".join((filename,'_{}'.format(batch))) for batch in range(nbatch)]
    
    # Making a list with the path of the files 
    files = [os.path.join(outdir, outname).replace("\\","/") for outname in names]
    
    # Create a DataFrame for the data
    df = pd.DataFrame()

    for path in files:
        # Load one of the files
        df_i = pd.read_csv(path)
        # Put it together with the main dataframe
        df = pd.concat([df, df_i])
    
    return df

def generator(filename, chunksize=512,total_size=1100000):
    """Generates iteratively a DataFrame from a large .h5 file

    If the arguments `chunksize` and `total_size` is not given in, 
    default values are used.

    Parameters
    ----------
    filename : str
        Path of the file

    chunksize : int
        Number of rows to generate (default is 512)

    total_size : int
        Total number of rows of the file (default is 1100000)

    Returns
    ------
    Pandas Dataframe
        A DataFrame with part of the data.
    """
    # Starting the counter
    m = 0
    
    while True:

        yield pd.read_hdf(filename,start=m*chunksize, stop=(m+1)*chunksize)

        m+=1
        # If we read all the data, reset the counter
        if (m+1)*chunksize > total_size:
            m=0

def ascii_column(path_label, column_name=2100):
    """Returns a DataFrame from an ASCII file of one column.

    If the argument `column_name` is not passed in, the default value is used.

    Parameters
    ----------
    path_label : str
        Path of the ASCII file

    column_name : str
        Name of the column (default is 2100)

    Returns
    ------
    df: Dataframe
        A one column DataFrame.
    """
    # Reading the archive
    with open(path_label, 'r') as f:
        data = f.read()
    # Converting it to a DataFrame
    df = pd.read_csv(StringIO(data), header=None, names=[column_name])

    return df

def separate_data(df, label='label', standarize=True):
    """Separates the features from the label and standarizes the features.

    If the argument `label` is not passed in, the default value is used.

    Parameters
    ----------
    df : DataFrame
        DataFrame with the data and the labels

    label : str
        Names of the column with the label (default is 'label')

    standarize: bool
        If True, standarizes the features (default is True)

    Returns
    ------
    X : DataFrame
        DataFrame with the features
    y : Serie
        Serie with the labels 
    """

    # Getting the DataFrame with the features
    X = df.drop(label, axis=1)
    
    # Standarizing the data 
    if standarize==True:
        for column in list(X.columns):
            feature = np.array(X[column]).reshape(-1,1)
            scaler = MinMaxScaler()
            scaler.fit(feature)
            feature_scaled = scaler.transform(feature)
            X[column] = feature_scaled.reshape(1,-1)[0]

    # Getting the serie wiht the labels
    y = df[label]
    
    return X, y

def save_df(outname, outdir, df):
    """Generates a csv from a DataFrame even if the given path does not exist.

    Parameters
    ----------
    outname : str
        Name of the generated file

    outdir : str
        Path where the file will be saved

    df : DataFrame
        DataFrame to save

    Returns
    ------
    File
        csv file (oudir/outname.csv)
    """
    
    # If the path does not exists, creates it
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    path = os.path.join(outdir, outname).replace("\\","/")   
    df.to_csv(path, sep=',', index=False)