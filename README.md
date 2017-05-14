# Machine Learning Project: An Exploratory Analysis of Using Machine Learning Algorithms to Predict Learning Ability in Mice with Down Syndrome

Eliana Marostica & Alexandria Guo
NOTE: had to reduce decimal points per data entry number

DATA SET
Download modified data/feature files:
  - protein_features.csv (https://drive.google.com/open?id=0B0fkiTnzLsi5eEp1dHRpRGw2cDg)
  - data.csv (https://drive.google.com/open?id=0B0fkiTnzLsi5bVVqemdKQzBhd3c)

FILES

data_loading.py
  - loads in the data from a csv file
  - splits the data into training, development, and testing
  - deals with missing data in three ways
  - normalizes and standardizes the data
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
  
