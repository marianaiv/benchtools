# Notebooks
Here are the notebooks with data processing and analysis using the benchtools package.

The notebooks are named following these rules:
- The first number refers to the order they were created.
- The second number is the version of the notebook. This will be increased only when new code is added.
- The name, separated by "-", has a general idea of the content.

Here is a brief description for the content of each one:

- **00.0-pre-processing-BB1** and **00.0-pre-processing-RD**: First exploration for the clustering of the black box 1 (BB1) and R&D dataset, respectively.
- **01.0-data-exploration**: Analysis of the data before and after clustering the jets. Calculation and plots of the variable's distributions and some correlations.
- **02.0-all-distribution-correlation-plots**: Plots for the distribution and correlation between all the variables, separated by background and signal.
- **03.0-decision-tree**: First use of a ML algorithm with the clustered data. A simple decision tree.
- **04.0-comparison-supervised-algorithms**: Training and predictions using multiple supervised algorithms from sklearn. First notebook with the calculation of performance metrics and plots.
- **05.0-GBC-classification**: Classification using Gradient Boosting Classifier (GBC). Calculation of performance metrics and comparison of the real distribution of the variables with the distributions obtained with the classifier.
- **06.0-GBC-overfitting**: An overfitting review for the classification made with GBC.
- **07.0-tensorflow-classificator**: Use of a simple sequential classifier made with tensorflow. Calculation Calculation of performance metrics and comparison of the real distribution of the variables with the distributions obtained with the classifier.
- **08.0-compararison-unsupervised-algorithms**: Training and predictions using multiple unsupervised algorithms from sklearn. Calculation of performance metrics and plots.
- **09.0-Kmeans-classification**: Classification using K-Means classifiier. Calculation of performance metrics and comparison of the real distribution of the variables with the distributions obtained with the classifier.