Project Proposal
Eliana Marostica and Alexandria Guo

Data Set: http://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression#

This dataset of protein expression in mice was generated based on experimentation in trisomic and healthy mice to determine the impact of memantine on protein expression levels. Existing work on this dataset (see links below) analyzes protein impact using an unsupervised clustering method known as Self Organizing Maps (SOM). We will be using a different approach to analyzing protein impact, while also looking at which ways are most effective in handling missing data values.
The mice are categorized in three ways--trisomic vs. control, context-shock vs. shock-context, and memantine vs. saline--and we propose adding a fourth categorization, learning vs. non-learning to these classifications. Instead of treating these as 8 (or 16 different classes), we will run 4 logistic regressions (supervised, binary linear classifiers) and analyze the most “impactful” genes from the largest component magnitudes of the hyperplane weight vectors. In examining our results (and mean squared error to verify the reliability and utility of our classifiers), we should be able to deduce which proteins and their expressions are affected under Down’s Syndrome, memantine-dosage, or different shock tactics, and in learning overall. We will also normalize the data, as well as experiment with different methods of handling missing data: ie, removing features/datapoints, using mean data points, statistical distribution sampling, collaborative filtering k-averages, etc. Overall, the primary purpose of our project is data analysis rather than prediction, specifically which features (proteins) are most influential in learning ability.

Existing Work:
  Self-Organizing Feature Maps Identify Proteins Critical to Learning in a Mouse Model of Down Syndrome:  http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0129126
  Protein Dynamics Associated with Failed and Rescued Learning in the Ts65Dn Mouse Model of Down Syndrome:  http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0119491

By Milestone 3, we will have: 
  written our logistic regression function
  written an expression to normalize our data (will have to normalize the protein expression values)
  completed preliminary runs of data through all 4 experiments; have made predictions and determined mean squared errors
  tested 1 or 2 alternative methods of handling missing data.
At minimum, by May 15th, we will have:
  choosen optimum missing data strategy (and other fine-tuning)
  determined which proteins had the most impact in each experiment
  completed data analysis of weight vectors, plus research into biological context
  Ideally, our logistic regression algorithm will accurately predict the labels, and we will be able to determine the proteins most impactful in learning.
