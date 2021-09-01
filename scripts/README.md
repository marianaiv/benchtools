# Scripts

Este es el archivo README para el subdirectorio de Scripts. Para el archivo README del repositorio principal siga [este link](https://github.com/marianaiv/benchmark_clalgoritmos/blob/main/README.md).

En este subdirectorio se encuentran los scripts donde:

- Se pre-procesan datos

## Instrucciones para correr scripts

### pre-procesamiento.py

   <code>python pre-procesamiento.py --RD --outname datosRD --nbatch 3</code>
   
   Las opciones para correrlo:
   
   * <code>--RD</code> : para utilizar el dataset RD
   * <code>--BB1</code> : para utilizar el dataset de la Black Box 1
   * <code>--dir</code> : para utilizar otros datos (en caso de no usar --RD), requiere un path de los datos 
   * <code>--dir</code> : para utilizar otros datos (en caso de no usar --RD), requiere un path de los labels 
   * <code>--outname</code>: nombre del archivo a generar
   * <code>--nbatch</code>: número de archivos de 51200 filas a generar
