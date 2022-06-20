# Benchtools

This is the README for the code of the *benchtools* package. For the README of the main repository follow [this link](https://github.com/marianaiv/benchmark_clalgoritmos/blob/main/README.md).

This file is constructed as follows:

- :bar_chart:**src**: files with functions developed to manage, pre-process and analize simulated data for multijet events.
- :small_orange_diamond:**scripts**: Pipeline for the benchmarking of classification algorithms and scripts to transform the classification of UCluster and GAN-AE into a classifier object.

## src
The `src` folder has the following modules:
- *clustering*: functions to cluster jets using pyjet.
- *datatools*: functions to manage big datasets and data related to the LHCO 2020.
- *metrictools*: functions to calculate performance metrics for binary classification algorithms.
- *plotools*: functions to plot the pre-processed data separating background from signal: distributions, correlations, etc.
- *substructure*: functions to calculate substructure variables from jets.
## scripts
The `scripts` folders contains the following:
### Run.py (pipeline)
Instructions to use the pipeline can be found in the ´[README file of the main repository](https://github.com/marianaiv/benchmark_clalgoritmos/blob/main/README.md).
### GAN-AE_data
To obtain the results of GAN-AE follow the instructions [here](https://github.com/marianaiv/GAN-AE_LHCOlympics). Save them on the `data` folder named `RnD_distances.h5` or `BB1_distances.h5` if R&D or BB1 datasets, respectively. With other number of boxes it will work too.

To use the script (assuming you are in this folder in the command line):
```
python GAN-AE_data.py [--RD]
```
where *RD* indicates to use the R&D file. Otherwise it will use the BB1 file. The number of box can be configured using `--box N` with N an integer, but the file has to be named `BBN_distances.h5`
### UCluster_data
To obtain the results of GAN-AE follow the instructions [here](https://github.com/marianaiv/UCluster). Save them on the `data` folder named `full_data_full_training_RD.h5` or `full_data_full_training_BB1.h5` if R&D or BB1 datasets, respectively. With other number of boxes it will work too. 

To use the script (assuming you are in this folder in the command line):
```
python UCluster_data.py [--RD]
```
where *RD* indicates to use the R&D file. Otherwise it will use the BB1 file. The number of box can be configured using `--box N` with N an integer, but the file has to be named `full_data_full_training_BBN.h5`
