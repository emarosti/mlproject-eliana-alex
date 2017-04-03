Link to download data: https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression

We had to manipulate the data a bit in excel, filling in empty cells with values of 0 and converting the different classes
into 0 and 1 binary representations in order to match the formating required for working with numpy arrays. We tried 
different approaches, but this seemed to be the best since all values had to be in float format and no protein expression
values were zero.

In data_loading.py we created a load function that imports the data and returns a tuple consisting of 
(1) An array of the data points and their protein expression values
(2) An array of the data points and their labels.
This is almost identical to how we've been formatting data in our problem sets. The only difference is that we have more than 
just one label. This formatting will allow us to more easily run our logistic regression code on different labels depending on 
whether we're testing for trisomy, teaching method, memantine use, or learning.

We then created functions to split the data into training and testing 80:20 using the StratifiedShuffleSplit function in
the sklearn package.
