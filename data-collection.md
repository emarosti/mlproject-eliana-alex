Link to download data: https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression

We had to first manipulate the data a bit in Excel, filling in empty cells with
values of 0.0 and converting the different class labels into binary representations
(0.0 or 1.0) in order to match the type requirements for working with numpy ndarrays.
We tried different approaches, but this seemed to be the best, since all values
had to be in float format and no protein expression values were already zero.

In data_loading.py we created a load function that imports the data and returns
a tuple consisting of:
(1) An array of the data points and their protein expression values
(2) An array of the data points and their labels.
This is almost identical to how we've been formatting data in our problem sets.
The only difference is that we have more than just one label. This formatting will
allow us to more easily run our logistic regression code on different labels depending
on whether we're separating data based on trisomy, teaching method, memantine use,
or learning.

We then created functions to split the data into training and testing sets (80:20
respectively) using the StratifiedShuffleSplit function in the sklearn package.
Although we currently have 4 different labeling systems/division, we used the 8-class
label to stratify the data, to ensure equal spread. The selected sets are then
saved as csv files--currently, only one split is being run, but the method can be
expanded for cross-validation later.
