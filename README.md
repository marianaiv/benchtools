<h1 align="center"> :sparkles: Benchtools :sparkles: </h1>

> Repository of the tools developed for the [final degree project](https://github.com/marianaiv/tesis_grado_UCV) of Mariana Vivas.   

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## **Table of Contents:**

- [About the project](#about_project)
- [About the repository](#about_repo)
- [The `Benchtools` package](#benchtools)
- [Development and testing](#testing)
- [License](#license)

## About the project <a name="about_project"></a>

The tools developed in the `Benchtools` package are for a project that aims to study **machine learning techniques for the search of new physics in dijets events by the development of tools to assert the performance of different approximations**. These approximations include algorithms from package like sklearn and simple tensorflow models, to more complex algorithms like [UCluster](https://github.com/ViniciusMikuni/UCluster) and [GAN-AE](https://github.com/lovaslin/GAN-AE_LHCOlympics), which participated in the [LHC Olympics 2020](lhco2020.github.io/homepage/). 

The project itself is in Spanish and more information about it and the results obtained can be found in this [link](https://github.com/marianaiv/tesis_grado_UCV).

## About the repository <a name="about_repo"></a>

**The tools developed to compare the performance of the algorithms are in this repository**. 

Some of the tools that can be found here are for:
- The use and transformation of data as the one used in the LHC Olympics 2020.
- Clustering of jets.
- Calculation of physical variables related to jets.
- Functions for plots separating signal and background.
- Calculation of performace metrics for the algorithms.
- Functions for plots to compare the performance of the algorithms.

### Content of the repository <a name="content"></a>
The content of this repository is organized as follows:
* :computer: [benchtools](benchtools): Package with the tools described above, scripts to transform data and a pipeline to compare binary classification algorithms.
* :books: [notebooks](notebooks): Jupyter Notebooks with data exploration and analysis, use of different ML algorithms and anything related to the development of the benchtools package. In these notebooks are examples of how the functions of the package can be used.
* :bar_chart: [tests](tests): Code for testing the functions and the pipeline in the benchtools package.

More information about the content of each file can be found on the README file on each folder.

## The `Benchtools` package <a name="benchtools"></a>
> A Python package for benchmarking binary classification algorithms 
### Workflow
`Benchtools` works with input data of dijet events as the one published for the [LHC Olympics 2020](lhco2020.github.io/homepage/):
> [The data has] the following event selection: at least one anti-kT R = 1.0 jet with pseudorapidity |η| < 2.5 and transverse momentum pT > 1.2 TeV.   For each event, we provide a list of all hadrons (pT, η, φ, pT, η, φ, …) zero-padded up to 700 hadrons.

As the idea is to use it to compare models, some simple models are trained and used to get predictions in the pipeline. However, a .txt can be passed to the pipeline with a list of files that contains a classifier object with the true labes, scores, and predictions given by any external classifier. 

`Benchtools` process the data, train some models and compares them with externaly inserted ones using performance metrics. The pipeline follows these steps (figure below):
- **Input**: .h5 file with the data and .txt with a list of files, each with a classifier object.
- **Steps**
1. Pre-process the data by clustering the jets and calculating pT, m, η, φ, E, 𝜏12, n_hadrons for the two principal jets and mjj, ΔR12, and n_hadrons for the event.
2. Scale the data and train the following classifiers: random forest, gradiend boosting, quadratic discriminant analysis, multilayer perceptron, K-Means and a sequential tensorflow model. 
3. Save the trained models.
4. Get predictions and scores from each classifier.
5. Loads the list of classifier objects from the .txt file.
6. Compares the algorithms:
    - Calculating balanced accuracy, precision, F1 score, recall, log loss and plotting this in bar plots to compare the classifiers.
    - Plotting: signal efficiency vs. background rejection, inverse ROC, significance improvement, precision-recall
- **Output**: Folder with .png for all the plots and a .txt of a table with the calculated variables.

## Development and testing <a name="testing"></a>

`Benchtools` uses the [pytest](https://pypi.org/project/pytest/) library for automated functional testing of code 
development and integration. These [tests](tests) are run from the project directory using the command:

`pytest -s `

## License <a name="license"></a>

This software is licensed under the terms of the [GNU General Public License v3.0 (GNU GPLv3)](https://choosealicense.com/licenses/gpl-3.0/).
