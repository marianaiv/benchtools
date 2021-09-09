import pyjet as fj

def deltaR(x, y):
    """Calcula la distancia angular entre dos jets.

    Parametros
    ----------
    x : PseudoJet
        Jet 1

    y : PseudoJet
        Jet 2

    Devuelve
    ------
    float
        Distancia angular entre dos jets
    """
    return ((x.phi-y.phi)**2 + (x.eta-y.eta)**2)**0.5

def subjettiness(cndts, cnsts):
    """Calcula subjettiness de un jet.

    Parametros
    ----------
    cndts : list
        Constituyentes del jet

    cnsts : PseudoJet
        Exclusive jet

    Devuelve
    ------
    float
        Calculo de subjettiness
    """
    d0 = sum(c.pt for c in cnsts)
    ls = []
    for c in cnsts:
        dRs = [deltaR(c,cd) for cd in cndts]
        ls += [c.pt * min(dRs)]
    return sum(ls)/d0

def tau21(jet,subR=0.2):
    """Calcula tau21(N-subjettiness) de un jet.

    Parametros
    ----------
    jet: PseudoJet
        Jet resultado del clustering    

    Devuelve
    ------
    float
        Calculo de tau21
    """
    jet_substruct_features = {}        
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
    except ZeroDivisionError:
        return 0