"""
Test clustering
"""
import pandas as pd
import os
from benchtools.src.clustering import jets, event_features, cluster_events, build_features
from benchtools.src.datatools import save_df

def test_jets():
    event = pd.read_hdf("../data/events_anomalydetection_tiny.h5", stop=1)
    event_as_serie = event.iloc[0]
    jets_first_event = jets(event_as_serie, R = 1.0, p = -1, minpt=20)
    
    # checking there are two jets:
    assert len(jets_first_event)==2

    # checking we get the pt for each jet
    assert jets_first_event[0].pt==1286.7276854490954
    assert jets_first_event[1].pt==1283.2207326902046

    # checking that changing parameters, changes output
    assert jets(event_as_serie, R = 1.0, p = -1, minpt=20) != jets(event_as_serie, R = 1.0, p = 1, minpt=20)

def test_event_features():
    event = pd.read_hdf("../data/events_anomalydetection_tiny.h5", stop=1)
    event_as_serie = event.iloc[0]
    jets_first_event = jets(event_as_serie, R = 1.0, p = -1, minpt=20)
    df = event_features(event_as_serie, jets_first_event)
    
    # checking columns
    columns=['pT_j1', 'm_j1', 'eta_j1', 'phi_j1', 'E_j1', 'tau_21_j1', 'nhadrons_j1',
            'pT_j2', 'm_j2', 'eta_j2', 'phi_j2', 'E_j2', 'tau_21_j2', 'nhadrons_j2',
            'm_jj', 'deltaR_j12','n_hadrons', 'label']
    assert df.columns.tolist() == columns

    # checking pt
    assert df.loc[0,'pT_j1']==1286.7276854490954
    assert df.loc[0,'pT_j2']==1283.2207326902046

    # checking for NaN values
    assert df.isnull().values.any() == False

def test_cluster_events():
    df_events = pd.read_hdf("../data/events_anomalydetection_tiny.h5")
    df_features = cluster_events(df_events)

    # cheking type
    assert type(df_features) == pd.core.frame.DataFrame

    # checking length 
    assert df_events.shape[0] == df_features.shape[0]

    # checking for NaN values
    df_features.isnull().values.any() == False

def test_build_features():
    
    build_features(path_data="../data/events_anomalydetection_tiny.h5",nbatch=1, outname='test_building', chunksize=10)
    df = pd.read_csv('../data/test_building_0.csv')
    
    # checking it has 10 rows
    assert df.shape[0]==10

    # checking expected columns
    columns = ['pT_j1','m_j1','eta_j1','phi_j1','E_j1','tau_21_j1','nhadrons_j1',
    'pT_j2','m_j2','eta_j2','phi_j2','E_j2','tau_21_j2','nhadrons_j2',
    'm_jj','deltaR_j12','n_hadrons','label']
    assert df.columns.tolist() == columns

    # checking for NaN values
    df.isnull().values.any() == False
    
    # deleting the file so it's created every time for the test
    if os.path.exists('../data/test_building_0.csv'):
        os.remove('../data/test_building_0.csv')