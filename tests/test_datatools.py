"""
Test datatools
"""
import pandas as pd
import numpy as np
import os

from benchtools.src.clustering import build_features
from benchtools.src.datatools import read_multifiles, ascii_column, separate_data, save_df


def test_read_multifiles():
  
    # Files created for testing with the commented code below
    #build_features(path_data="data/events_anomalydetection_tiny.h5",nbatch=2, outname='test_file', chunksize=100)
    #df_random = pd.DataFrame(np.random.randint(0,10,size=(10, 4)), columns=list('ABCD'))
    #df_random.to_csv('data/test_multifiles_0.csv',index=False)
    #df_random.to_csv('data/test_multifiles_1.csv',index=False)
    
    # test lenght:
    df = read_multifiles('test_multifiles', nbatch=2, outdir = 'data')
    assert df.shape[0]==10*2

    # check for duplicated indexes
    assert df.index.duplicated().all()==False

#def test_generator():

def test_ascii_column():
    df = ascii_column(path_label='data/test_file.masterkey')
    # checking the length
    assert df.shape == (40,1)
    # checking that it only has 0 and 1
    assert df[2100].isin([0,1]).all() == True

def test_separate_data():
    df = pd.read_csv('data/test_file.csv')
    X, y = separate_data(df)
    
    # check y only has 0 or 1
    assert y.isin([0,1]).all() == True

    # check columns of X
    assert X.columns.tolist() == df.drop('label', axis=1).columns.tolist()

    # check the rows of both df is the same
    assert X.shape[0]==y.shape[0]

def test_save_df():
    df = pd.read_csv('data/test_file.csv')
    save_df(outname='test', outdir='data/test_directory', df=df)
    
    # check for creation of the directory an file
    assert os.path.exists('data/test_directory/test.csv')==True

    # removing the files created for the test
    if os.path.exists('data/test_directory'):
        os.remove('data/test_directory/test.csv')
        os.rmdir('data/test_directory')