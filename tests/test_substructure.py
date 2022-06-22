"""
Test substructure

Disclaimer: the tests for 'subjettiness' and 'tau21' can fail if the function 'jets'
from benchtools.src.clustering is failing, and 'tau21' can fail if 'subjettines' is 
failing
"""
import pandas as pd
import pyjet as fj
from benchtools.src.clustering import jets
from benchtools.src.substructure import deltaR, invariantmass, subjettiness, tau21

class fakejet:
  def __init__(self, pt, eta, phi, mass, E, px=None, py=None, pz=None):
    self.pt = pt
    self.eta = eta
    self.phi = phi
    self.mass = mass
    self.e = E
    self.px = px
    self.py = py
    self.pz = pz

def test_deltaR():
    # defining jets
    null_jet = fakejet(0, 0, 0, 0, 0)
    jet_one = fakejet(1286.728, 0.186, -2.764, 106.912, 1313.290)
    jet_two = fakejet(1283.221, 0.065, 0.394, 63.164, 1287.482)

    # testing the calculation with all zeros
    assert deltaR(null_jet,null_jet) == 0

    # testing the value of two real jets from the first event
    assert deltaR(jet_one, jet_two) == 3.1603172309121117

    # testing the value with one real jet and one null jet
    assert deltaR(null_jet, jet_one) == 2.770251252143025

    # testing with two equal jets
    assert deltaR(jet_two, jet_two) == 0

def test_subjettiness():
    # creating real jets from the first event
    event = pd.read_hdf('data/events_anomalydetection_tiny.h5', stop=1)
    event_as_serie = event.iloc[0]
    jets_event = jets(event_as_serie, R = 1.0, p = -1, minpt=20)
    # getting constituents 
    cnsts = jets_event[0].constituents()
    # clustering inside the jet
    seq = fj.cluster(jets_event[0], R=0.2, algo='kt')
    cndts1 = seq.exclusive_jets(1)

    assert subjettiness(cndts1, cnsts) == 0.020976077764451328

def test_tau21():
    # creating real jets from the first event
    event = pd.read_hdf('data/events_anomalydetection_tiny.h5', stop=1)
    event_as_serie = event.iloc[0]
    jets_event = jets(event_as_serie, R = 1.0, p = -1, minpt=20)
    # getting constituents 
    cnsts = jets_event[0].constituents()
    # clustering inside the jet
    seq = fj.cluster(jets_event[0], R=0.2, algo='kt')
    # one jet
    cndts1 = seq.exclusive_jets(1)
    tau1 = subjettiness(cndts1, cnsts)
    # two jets
    if (len(cnsts)>1):
        cndts2 = seq.exclusive_jets(2)
        tau2 = subjettiness(cndts2, cnsts)

    assert tau21(jets_event[0]) == 0.6246590884270066


def test_invariantmass():
    # defining jets
    null_jet = fakejet(0, 0, 0, 0, 0, 0, 0, 0)
    jet_one = fakejet(1286.728, 0.186, -2.764, 106.912, 1313.290, -1195.931, -474.783, 240.070)
    jet_two = fakejet(1283.221, 0.065, 0.394, 63.164, 1287.482, 1185.055, 492.240, 83.454)

    # testing the calculation with all zeros
    assert invariantmass(null_jet, null_jet) == 0
    
    # testing the value for two real jets
    assert invariantmass(jet_one, jet_two) == 2580.489136420264

    # testing for one null and one real
    assert invariantmass(jet_one, null_jet) == 106.8979202323398