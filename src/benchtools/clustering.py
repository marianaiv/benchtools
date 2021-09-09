import pyjet as fj
from subestructura import deltaR, tau21
import pandas as pd
import numpy as np

def jets(evento, R = 1.0, p = -1, ptmin=20): 
    """Genera una lista de jets dado un evento.

    Si el argumento `n_hadrones`, `R`, `p` y `ptmin` no es pasado, 
    se utilizan valores predeterminados.

    Parametros
    ----------
    evento : serie
        Hadrones del evento

    R : int
        Tamaño del radio a utilizar para el clustering (por defecto es 1)

    p : int
        Algoritmo a utilizar para el clustering (por defecto es -1 o kt)
    
    ptmin : int
        pT mínimo de los jets a listar (por defecto es 20)

    Devuelve
    ------
    lista
        Lista de objetos PseudoJets, o jets con ptmin clusterizados para el evento
    """
    
    datos_ss = evento.iloc[:-1]
    n_hadrones = int(evento.shape[0]/3)

    pseudojets_input = np.zeros(len([data for data in datos_ss.iloc[::3] if data > 0]), dtype= fj.DTYPE_PTEPM) 

    for hadron in range(n_hadrones):
        if (datos_ss.iloc[hadron*3] > 0): ## si pT > 0 

            ## Llenamos el arreglo con pT, eta y phi de cada "partícula"
            pseudojets_input[hadron]['pT'] = datos_ss.iloc[hadron*3] 
            pseudojets_input[hadron]['eta'] = datos_ss.iloc[hadron*3+1]
            pseudojets_input[hadron]['phi'] = datos_ss.iloc[hadron*3+2]

            pass
        pass

    ## Devuelve una "ClusterSequence" (un tipo de lista de pyjet)
    secuencia = fj.cluster(pseudojets_input, R = 1.0, p = -1) 

    ## Con inclusive_jets accedemos a todos los jets que fueron clusterizados
    ## y filtramos los que tienen pT mayor que 20.
    ## Hacemos una lista con objetos PseudoJet
    jets = secuencia.inclusive_jets(ptmin)
    
    return jets

def variables(evento, jets):
    """Genera un dataframe de variables dado un evento.

    Parametros
    ----------
    evento : serie
        Hadrones del evento

    jets : lista
        Lista de jets clusterizados para el evento

    Devuelve
    ------
    DataFrame
        Con pT, mj, eta, phi, E y tau para los dos jets principales,
        y deltaR, mjj y nro. de hadrones para el evento.
    """
# Extraemos las variables de interes del jet principal
    pT_j1 = jets[0].pt
    m_j1 = np.abs(jets[0].mass)
    eta_j1 = jets[0].eta
    phi_j1 = jets[0].phi
    E_j1 = jets[0].e
    tau_21_j1= tau21(jets[0])
    nhadrones_j1 = len(jets[0].constituents())

    # Intentamos extraer las variables del jet secundario
    try:
        pT_j2 = jets[1].pt
        m_j2 = np.abs(jets[1].mass)
        eta_j2 = jets[1].eta
        phi_j2 = jets[1].phi
        E_j2 = jets[1].e
        tau_21_j2= tau21(jets[1])
        nhadrones_j2 = len(jets[1].constituents())

    # Si no hay jet secundario colocamos ceros
    except IndexError:
        pT_j2 = 0
        m_j2 = 0
        eta_j2 = 0
        phi_j2 = 0
        E_j2 = 0
        tau_21_j2 = 0
        nhadrones_j2 = 0

    # Calculamos otras variables
    deltaR_j12 = deltaR(jets[0], jets[1])
    mjj = m_j1 + m_j2
    n_hadrones = evento.iloc[:-1].astype(bool).sum(axis=0)/3
    label = evento.iloc[-1]

    # Agregamos todo al dataframe
    entry = pd.DataFrame([[pT_j1, m_j1, eta_j1, phi_j1, E_j1, tau_21_j1, nhadrones_j1, 
                            pT_j2, m_j2, eta_j2, phi_j2, E_j2, tau_21_j2, nhadrones_j2,
                            mjj,deltaR_j12, n_hadrones, label]],
                        columns=['pT_j1', 'm_j1', 'eta_j1', 'phi_j1', 'E_j1', 'tau_21_j1', 'nhadrones_j1',
                            'pT_j2', 'm_j2', 'eta_j2', 'phi_j2', 'E_j2', 'tau_21_j2', 'nhadrones_j2',
                            'm_jj', 'deltaR_j12','n_hadrones', 'label'])
    
    return entry