import pyjet as fj

def deltaR(x, y):
    """Calculates the angular distance between two jets.

    Parameters
    ----------
    x : PseudoJet
        Jet 1

    y : PseudoJet
        Jet 2

    Returns
    ------
    float
        Angular distance between two jets
    """
    return ((x.phi-y.phi)**2 + (x.eta-y.eta)**2)**0.5

def subjettiness(cndts, cnsts):
    """Calculates subjettiness of one jet.

    Parameters
    ----------
    cndts : list
        Jet constituents

    cnsts : PseudoJet
        Exclusive jets

    Devuelve
    ------
    float
        Value for subjettiness
    """
    d0 = sum(c.pt for c in cnsts)
    ls = []
    for c in cnsts:
        dRs = [deltaR(c,cd) for cd in cndts]
        ls += [c.pt * min(dRs)]
    return sum(ls)/d0

def tau21(jet,subR=0.2):
    """Calculates tau21(N-subjettiness) of a jet.

    Parameters
    ----------
    jet : PseudoJet
        Resulting jet from the clustering 
    subR : float  

    Devuelve
    ------
    float
        Value for tau21
    """       
    seq = fj.cluster(jet, R=subR, algo='kt')
    cnsts = jet.constituents()
    cndts1 = seq.exclusive_jets(1)
    tau1 = subjettiness(cndts1, cnsts)
    if (len(cnsts)>1):
        cndts2 = seq.exclusive_jets(2)
        tau2 = subjettiness(cndts2, cnsts)
    else: 
        tau2 = 0

    try:     
        return tau2/tau1
    # I added this because it was giving me an error
    except ZeroDivisionError:
        return 0

def invariantmass(jet1, jet2):
    """Calculates the invariant mass from two jets.

    Parameters
    ----------
    jet1 : PseudoJet
        Resulting jet from the clustering 
    jet2 : PseudoJet
        Resulting jet from the clustering  

    Devuelve
    ------
    float
        Value for the invariant mass mjj
    """
    E = jet1.e + jet2.e
        
    px = jet1.px + jet2.px 
    py = jet1.py + jet2.py
    pz = jet1.pz + jet2.pz

    mjj = (E**2-px**2-py**2-pz**2)**0.5
    return mjj