import pandas as pd
from io import StringIO
import os.path                          # Manejo de directorios
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None  # default='warn'


def generador(filename, chunksize=512,total_size=1100000):
    """Genera un dataframe de pandas utilizando parte de los datos de entrada,
    manteniendo una cuenta para poder generar todos los datos iterativamente.

    Si el argumento `chunksize` y `total_size` no es pasado, 
    se utilizan valores predeterminados.

    Parametros
    ----------
    filename : str
        Dirección del conjunto de datos .h5

    chunksize : int
        Tamaño del conjunto de datos a generar (por defecto es 512)

    total_size : int
        Numero de filas del archivo (por defecto es 1100000)

    Devuelve
    ------
    Pandas Dataframe
        Un dataframe con parte del conjunto de datos
    """
    m = 0
    
    while True:

        yield pd.read_hdf(filename,start=m*chunksize, stop=(m+1)*chunksize)

        m+=1
        if (m+1)*chunksize > total_size:
            m=0

def leer_labels(path_label, column_name=2100):
    """Lee el archivo ASCII que contiene los labels de los eventos: si son señal o fondo.

    Si el argumento `column_name` no es pasado, se utiliza un nombre predeterminados.

    Parametros
    ----------
    path_label : str
        Dirección del archivo ASCII con los labels

    column_name : str
        Nombre de la columna (por defecto es 2100)

    Devuelve
    ------
    Pandas Dataframe
        Un dataframe con parte los labels de los eventos
    """
    # leemos el archivo con el key. Es un archivo ASCII donde cada linea corresponde a la información de señal o fondo
    with open(path_label, 'r') as f:
        data = f.read()
    # Lo convertimos en dataframe
    df = pd.read_csv(StringIO(data), header=None)
    # Le cambiamos el nombre a la columna para que al concatenar con los datos no tenga el mismo nombre que otra columna
    df = df.rename(columns={0: column_name})
    
    return df

def guardar(outname, outdir, df):
    """Genera un archivo csv a partir de un DataFrame.

    Parametros
    ----------
    outname : str
        Nombre del archivo a generar

    outdir : str
        Dirección de la carpeta donde se va a guardar el archivo

    df : DataFrame
        DataFrame a guardar

    Devuelve
    ------
    Archivo
        Archivo csv (oudir/outname.csv)
    """
    
    # Si no existe la carpeta, la crea
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    path = os.path.join(outdir, outname).replace("\\","/")   
    df.to_csv(path, sep=',', index=False)