# Machine Learning Project: An Exploratory Analysis of Using Machine Learning Algorithms to Predict Learning Ability in Mice with Down Syndrome

Eliana Marostica & Alexandria Guo

DATA SET
Download modified data/feature files:
<<<<<<< HEAD
  protein_features.csv (https://drive.google.com/open?id=0B0fkiTnzLsi5eEp1dHRpRGw2cDg)
  data.csv (https://drive.google.com/open?id=0B0fkiTnzLsi5bVVqemdKQzBhd3c)
Modifications:
  - made number of decimal points per data entry number consistent
  - replaced all missing values with 0.0
  - converted string class labels to 0.0/1.0 (float)
  - added learning class labels (0.0/1.0) and treatment group labels (0.0-7.0)
=======
  - protein_features.csv (https://drive.google.com/open?id=0B0fkiTnzLsi5eEp1dHRpRGw2cDg)
  - data.csv (https://drive.google.com/open?id=0B0fkiTnzLsi5bVVqemdKQzBhd3c)
>>>>>>> 8544d9a1d58cbb700ec9e34b73b49d7fcdee0ae8

FILES
data_loading.py
  - loads in the data from a csv file
  - splits the data into training, development, and testing sets
  - deals with missing data in three ways
  - normalizes or standardizes the data
  - loads in the features (proteins) from a csv file for future feature analysis

linear_classifier.py
  - contains code for logistic regression, SVM, and SVC + kernel
  - contains code for determining accuracy of predictions
  - contains code for determining the most significant features (proteins) when determining these predictions

random_forest.py
  - contains code for a Random Forest classifier
  - contains code for determining accuracy of predictions
  - contains code for determining the most significant features (proteins) when determining these predictions

knn.py
  - contains code for a k Nearest Neighbors classifier
  - contains code for determining accuracy of predictions
