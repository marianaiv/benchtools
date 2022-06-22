# Notebooks
Here are the notebooks with data processing and analysis using the benchtools package.

The notebooks are named following these rules:
- The first number refers to the order they were created.
- The second number is the version of the notebook. This will be increased only when new code is added.
- The name, separated by "-", has a general idea of the content.
## Content
Here is a brief description for what you will find in each one:

- :books: **00.0-pre-processing-BB1** and **00.0-pre-processing-RD**: First exploration about clustering using black box 1 (BB1) and R&D dataset.   

- :books: **01.0-data-exploration**: Data analysis before and after clustering the jets. Calculations and plots of the variable's distributions and some correlations.   

- :books: **02.0-all-distribution-correlation-plots**: Plots for the distribution of all the variables and the correlations between them, separated as background and signal.    

- :books: **03.0-decision-tree**: First use of a ML algorithm with the clustered data. A simple decision tree.   

- :books: **04.0-comparison-supervised-algorithms**: Classification using multiple supervised algorithms from sklearn. First notebook with the calculation of performance metrics and plots.   

- :books: **05.0-GBC-classification**: Classification using Gradient Boosting Classifier (GBC). Calculation of performance metrics and comparison of the real distribution of the variables with the distributions obtained with the classifier.   

- :books: **06.0-GBC-overfitting**: An overfitting review for the classification made with GBC.   

- :books: **07.0-tensorflow-classificator**: Use of a simple sequential classifier made with tensorflow. Calculation of performance metrics and comparison between the real distribution of the variables and the distributions obtained with the classifier.   

- :books: **08.0-compararison-unsupervised-algorithms**: Classification using multiple unsupervised algorithms from sklearn. Calculation of performance metrics and plots.   

- :books: **09.0-Kmeans-classification**: Classification using K-Means classifiier. Calculation of performance metrics and comparison between the real distribution of the variables and the distributions obtained with the classifier.   

- :books: **10.0-UCluster-data**: Exploring pre-processed data from UCluster and the classifications made with this algorithm.

- :books: **11.0-GAN-AE-data**: Exploring pre-processed data from GAN-AE and the classification made with this algorithm.

- :books: **13.0-scalers-comparison**: Comparison of the classification using different scalers: MinMaxScaler, StandardScaler and RobustScaler.

- :books: **14.0-dimension-reduction-comparison**: Comparison of the classification using different dimensionality reduction techniques: PCA, SCD, LDA.

- :books: **15.0-pipeline-exploration**: Code writting for the pipeline. Exploration of options.

- :books: **16.0-pipeline-KMeans-performance**: Issue with KMeans classification, solved by choosing the background label to the majoritary class predicted.