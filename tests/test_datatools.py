"""
Test datatools
"""
import pandas as pd
import os

from benchtools.src.clustering import build_features
from benchtools.src.datatools import read_multifiles, generator, ascii_column, separate_data, save_df


def test_read_multifiles():
    
    #build_features(path_data="../data/events_anomalydetection_tiny.h5",nbatch=2, outname='test_file', chunksize=100)
    
    # test lenght:
    df = pd.read_csv('../data/test_file.csv')
    assert df.shape[0]==100*2

    # check for duplicated indexes
    assert df.index.duplicated().all()==False

#def test_generator():

def test_ascii_column():
    df = ascii_column(path_label='../data/test_file.masterkey')
    # checking the length
    assert df.shape == (40,1)
    # checking that it only has 0 and 1
    df[2100].isin([0,1]).all() == True

def test_separate_data():
    df = pd.read_csv('../data/test_file.csv')
    X, y = separate_data(df)
    
    # check y only has 0 or 1
    assert y.isin([0,1]).all() == True

    # check columns of X
    assert X.columns.tolist() == df.drop('label', axis=1).columns.tolist()

    # check the rows of both df is the same
    assert X.shape[0]==y.shape[0]

def test_save_df():
    df = pd.read_csv('../data/test_file.csv')
    save_df(outname='test', outdir='../data/test_directory', df=df)
    
    # check for creation of the directory an file
    assert os.path.exists('../data/test_directory/test.csv')==True

    # removing the files created for the test
    if os.path.exists('../data/test_directory'):
        os.remove('../data/test_directory/test.csv')
        os.rmdir('../data/test_directory')