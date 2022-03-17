# Benchmarking of classification algorithms

This project is part of the [final degree project](https://github.com/marianaiv/tesis_grado_UCV) of Mariana Vivas.

### About the project 

The project is about the **study of machine learning techniques for the search of new physics in dijets events by developing tools to assert the performance of different approximations**. These approximations include algorithms from package like sklearn and simple tensorflow models, to more complex algorithms like [UCluster](https://github.com/ViniciusMikuni/UCluster) and [GAN-AE](https://github.com/lovaslin/GAN-AE_LHCOlympics), which participated in the [LHC Olympics 2020](lhco2020.github.io/homepage/). The project itself is in Spanish, but for more information about it and the results obtained you can follow this [link](https://github.com/marianaiv/tesis_grado_UCV).

**The tools developed to compare the performance of the algorithms are in this repository**. Some of the tools that can be found here are for:
- The use and transformation of the type of data given in the LHC Olympics 2020.
- Calculation of physical variables related to jet events.
- Clustering of jet events.
- Functions for plots separating signal and background.
- Calculation of performace metrics for the algorithms.
- Functions for plots to compare the performance of the algorithms.

## Content
The content of this repository is organized as follows:
* :computer: [benchtools](benchtools): Package with the tools described above, scripts to transform data and a pipeline to compare binary classification algorithms.
* :books: [notebooks](notebooks): Jupyter Notebooks with data exploration and analysis, use of different ML algorithms and anything related to the development of the benchtools package. In these notebooks are examples of how the functions of the package can be used.
* :bar_chart: [tests](tests): Code for testing the functions and the pipeline in the benchtools package.

More information about the content of each file can be found on the README file on each folder.
