# Importamos las librerías a utilizar
#import h5py                                                               # Para manejar los archivos .h5
#import pyjet as fj                                                        # Clustering de los jets
#import numpy as np              
import pandas as pd                                                        # Manejo de tablas
#import os.path                                                            # Manejo de directorios
from benchtools.subestructura import deltaR, tau21                                    # Calculo de variables 
from benchtools.clustering import jets, variables                          # Funciones para el clustering
from benchtools.datatools import generador, leer_labels, guardar
#from os import path                                                       # Manejo de paths
from tqdm import tqdm                                                      # Barra de progreso
from optparse import OptionParser

parser = OptionParser(usage="%prog [opt]  inputFiles")
parser.add_option("--RD", action="store_true", default=False, help="Usa el conjunto de datos RD")
parser.add_option("--BB1", action="store_true", default=False, help="Usa el conjunto de datos BB1")
parser.add_option("--dir", type="string", default= None, help="Path del conjunto de datos")
parser.add_option("--label", type="string", default= None, help="Path del label del conjunto de datos")
parser.add_option("--out", type="string", default="../data/", help="Carpeta para salvar los datos generados")
parser.add_option("--outname", type="string", default="None", help="Nombre del archivo para salvar los datos generados")
parser.add_option("--nbatch", type="int", default=3, help="Número de batches de la data a generar")

(flags, args) = parser.parse_args()

RD = flags.RD 
BB1 = flags.BB1

if RD:
    path_datos = "../../events_anomalydetection.h5"

elif BB1:
    path_datos = "../../events_LHCO2020_BlackBox1.h5"
    path_label = "../../events_LHCO2020_BlackBox1.masterkey"

else: 
    path_datos = flags.dir
    path_label = flags.label

fb = generador(path_datos,chunksize=512*100)

# Iniciamos el contador
batch_idx = 0

for batch in fb:
    # Definimos un dataframe para guardar las variables 
    df = pd.DataFrame(columns=['pT_j1', 'm_j1', 'eta_j1', 'phi_j1', 'E_j1', 'tau_21_j1', 'nhadrones_j1',
                            'pT_j2', 'm_j2', 'eta_j2', 'phi_j2', 'E_j2', 'tau_21_j2', 'nhadrones_j2',
                            'm_jj', 'deltaR_j12','n_hadrones', 'label'])
    
    if batch_idx < flags.nbatch:
        print('Parte {}/{}'.format(batch_idx+1, flags.nbatch))
    elif batch_idx == flags.nbatch:
        print('Pre-procesamiento terminado')
        break
        
    datos = batch
    if RD == False:
        label = leer_labels(path_label)
        datos = pd.concat([datos, label.iloc[datos.index]], axis=1)

    nro_eventos = datos.shape[0]

    for evento in tqdm(range(nro_eventos)):

        # Obtenemos los datos de un evento
        datos_evento = datos.iloc[evento,:]
        
        # Obtenemos los jets 
        jets_evento = jets(datos_evento, R = 1.0, p = -1, ptmin=20)
        
        # Calculamos las variables y las guardamos en un dataframe
        entry = variables(datos_evento, jets_evento)
        
        # Adjuntamos al dataframe inicial
        df = df.append(entry, sort=False)
    
    # Guardamos el dataframe como csv para cada batch  
    
    outname = "".join((flags.outname,'_{}')).format(batch_idx)
    outdir = '../data'
    guardar(outname, outdir, df)
    
    batch_idx += 1 