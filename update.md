In our proposal, we stated that by Milestone 3, we would have completed the following tasks:

-written our logistic regression function
-written an expression to normalize our data (will have to normalize the protein expression values)
-completed preliminary runs of data through all 4 experiments; have made predictions and determined mean squared errors
-tested 1 or 2 alternative methods of handling missing data

In the two weeks since Milestone 2, we have completed the following tasks:

-created an expression to fill in the missing values in our data with the average expression value for that protein (as per how existing work stated)
-normalized our data so that all values are between 0 and 1
-further normalized our data using the standardization strategy to center the data around 0*
-updated our data splitting code so that we have training, development, and testing sets
-written our logistic regression function
-completed preliminary runs of data through all 3 experiments
-have made predictions and determined *accuracies* (we had planned to calculate mean squared errors but now we have decided to use accuracies)


We are right on track with our original plan. In the weeks to come, we need to complete the following tasks:

-tested other missing data strategies
-choose our optimum missing data strategy
-implement gridsearch (tuning hyperparameters)
-determine the optimum linear classifier
-determined which proteins had the most impact in each experiment (implement "show_significant_features")
-completed data analysis of weight vectors, plus research into biological context
-determine the proteins most impactful in learning.


Difficulties We've Encountered:
*When using the standardization strategy to center our data around 0, we included the missing-turned-mean values in our calculations.
    Is this a bad idea? Should we exclude the missing values? We are not sure, because we think that including them would lower our
    standard deviation values for the proteins. Would this render them inaccurate? If we don't include them in the calculations, the
    data would still be centered around 0 when we reinclude them, correct?
