import pyjet as fj
from benchtools.substructure import deltaR, tau21
import pandas as pd
import numpy as np

def jets(event, R = 1.0, p = -1, minpt=20): 
    """Returns a list of jets given an event.

    If the argument `n_hadrones`, `R`, `p` and `minpt` arent't passed in, 
    the default values are used.

    Parameters
    ----------
    event : Serie
        Hadrons of the event 

    R : int
        Size of the radius for the clustering (default is 1)

    p : int
        Algorithm to use for the clustering (default os -1 or kt)
    
    minpt : int
        Minimum pT of the jets on the list (default is 20)

    Returns
    ------
    jets: list
        List of PseudoJets objects that represent the jets
        of the event
    """
    
    # No signal data
    nsdata = event.iloc[:-1]
    # Number of hadrons of the event
    n_hadrons = int(event.shape[0]/3)

    # Were the info of every hadron will go
    pseudojets_input = np.zeros(len([data for data in nsdata.iloc[::3] if data > 0]), dtype= fj.DTYPE_PTEPM) 

    for hadron in range(n_hadrons):
        if (nsdata.iloc[hadron*3] > 0): ## if pT > 0 

            ## Filling with pT, eta and phi of each hadron
            pseudojets_input[hadron]['pT'] = nsdata.iloc[hadron*3] 
            pseudojets_input[hadron]['eta'] = nsdata.iloc[hadron*3+1]
            pseudojets_input[hadron]['phi'] = nsdata.iloc[hadron*3+2]

            pass
        pass

    ## Returns a "ClusterSequence" (pyjets type of list)
    secuencia = fj.cluster(pseudojets_input, R = 1.0, p = -1) 

    ## With inclusive_jets we access all the jets that were clustered
    ## and we filter those with pT greater than 20.
    jets = secuencia.inclusive_jets(minpt)
    
    return jets

def features(event, jets):
    """Returns a DataFrame of calculated features given an event.

    Parameters
    ----------
    event : Serie
        Hadrons of the event 

    jets : list
        List of the clusterized jets for the event

    Returns
    ------
    entry: DataFrame
        A DataFrame with pT, mj, eta, phi, E y tau for the two principal jets,
        and deltaR, mjj y number of hadrons for the event.
    """
    # We extract the variables of interest from the main jet 
    pT_j1 = jets[0].pt
    m_j1 = np.abs(jets[0].mass)
    eta_j1 = jets[0].eta
    phi_j1 = jets[0].phi
    E_j1 = jets[0].e
    tau_21_j1= tau21(jets[0])
    nhadrons_j1 = len(jets[0].constituents())

    # We do the same if there is a secundary jet
    try:
        pT_j2 = jets[1].pt
        m_j2 = np.abs(jets[1].mass)
        eta_j2 = jets[1].eta
        phi_j2 = jets[1].phi
        E_j2 = jets[1].e
        tau_21_j2= tau21(jets[1])
        nhadrons_j2 = len(jets[1].constituents())

    # If there is not a secundary jet, all the variables will be zero
    except IndexError:
        pT_j2 = 0
        m_j2 = 0
        eta_j2 = 0
        phi_j2 = 0
        E_j2 = 0
        tau_21_j2 = 0
        nhadrons_j2 = 0

    # Calculating the general variables of the event
    deltaR_j12 = deltaR(jets[0], jets[1])
    mjj = m_j1 + m_j2
    n_hadrons = event.iloc[:-1].astype(bool).sum(axis=0)/3
    # Getting the label of the event: signal or background
    label = event.iloc[-1]

    # Adding everything to a dataframe
    entry = pd.DataFrame([[pT_j1, m_j1, eta_j1, phi_j1, E_j1, tau_21_j1, nhadrons_j1, 
                            pT_j2, m_j2, eta_j2, phi_j2, E_j2, tau_21_j2, nhadrons_j2,
                            mjj,deltaR_j12, n_hadrons, label]],
                        columns=['pT_j1', 'm_j1', 'eta_j1', 'phi_j1', 'E_j1', 'tau_21_j1', 'nhadrons_j1',
                            'pT_j2', 'm_j2', 'eta_j2', 'phi_j2', 'E_j2', 'tau_21_j2', 'nhadrons_j2',
                            'm_jj', 'deltaR_j12','n_hadrons', 'label'])
    
    return entry