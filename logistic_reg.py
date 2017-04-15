from __future__ import division  # floating point division
import time  # time utilities
import sys

"""Implement gradient ascent for logistic regression."""

import numpy as np
import data_loading
from numpy.linalg import norm

def sigmoid(x, w, b):
    """P(y=1|x, w, b) with the sigmoid (logistic) function"""
    wx_plus_b = w.dot(x)+b
    # handle overflow
    if wx_plus_b<-500:
        return 0
    else:
        return 1./(1+np.exp(-wx_plus_b))

# PART B: Write your code below
def logreg_train(trainX, trainY, maxiter, eta, alpha, batch_size=1):
    """Given training data points (trainX) and their labels (trainY -- each either 0 or +1),
    train a hyperplane represented by a weight vector w and bias b
    using logistic regression, for maxiter epochs using the eta learning rate
    and alpha L2 regularization weight.

    If eta is a number, use it as a constant learning rate.
    If it is the string 'step', start with a rate of 1, and reduce it by 1/2
    every 5 epochs. (This is called step decay.)

    If batch_size is 1, use stochastic gradient ascent, where the gradient is calculated
    at each training point and w and b updated accordingly.
    Otherwise, split the data into batches given by batch_size, and compute the gradient
    and make the update on each batch (mini-batch gradient ascent).

    Return a tuple consisting of
    (1) the weight vector
    (2) the bias
    (3) the training accuracy at the end
    """
    num_traindata, num_dims = trainX.shape
    trainY = np.array(trainY)
    if (eta == 'step'): learn_rate = 1
    else: learn_rate = eta

    weights = np.zeros(num_dims+1) # bias trick, p1
    trainX = np.concatenate((np.ones((num_traindata, 1)), trainX), axis=1) # bias trick, p2

    batch_indices = np.arange(-1, num_traindata, batch_size) # note which indices update occurs
    batch_indices = np.append(batch_indices, num_traindata-1) # always update on last run

    for epoch in range(maxiter):
        gradient = np.zeros(num_dims+1) # (re)initialize gradient
        error_count = 0 # (re)initialize error count
        update = 1 # ignore the first index -1 in batch_indices
        if (eta == 'step') and (epoch%5 == 0): learn_rate = learn_rate / 2 # step decay
        for index in range(num_traindata):
            likelihood = (trainY[index] - sigmoid(trainX[index,1:], weights[1:], weights[0])) # L(theta)
            if (abs(likelihood) >= 0.5): error_count += 1 # increment on incorrect prediction
            no_bias = weights[1:] * alpha # do not include bias in L2 regularization
            gradient += (likelihood * trainX[index,]) - (np.concatenate((np.zeros(1), no_bias))) # update gradient
            if (index == batch_indices[update]):
                weights += learn_rate * gradient # update weights
                gradient = np.zeros(num_dims+1) # reinitialize gradient
                update += 1 # check next index to batch update on
    return (weights[1:num_dims+1], weights[0], 1-(error_count/num_traindata))

def predict_confidence(w, b, testX):
    """return a vector of probabilities of the class y=1 for each data point x"""
    num_testdata, num_dims = testX.shape
    testY = np.zeros(num_testdata)
    for i in range(num_testdata): testY[i] = sigmoid(testX[i], w, b)
    return testY

def get_meansq_accuracy(testy, predictions):
    """return mean square accuracy of predictions"""
    accuracy = 0
    for i in range(len(testy)): accuracy += ((predictions[i] - testy[i])**2.0)
    return 1-((1/len(testy)) * accuracy)

def get_accuracy(testy, predictions):
    """return proportion of correct predictions"""
    # convert probabilities to 0 or 1 predictions first
    predictions = np.array(predictions)
    predictions = (predictions>0.5).astype(int)  # 1 if over 50%, 0 if not
    return 1-norm(predictions-testy, 0)/float(len(testy))

def show_significant_features(w, featurelist):
    wsorted = np.argsort(w)
    print 'Features predicting bill survived:', ', '.join(map(lambda i: featurelist[i], wsorted[:30]))
    print 'Features predicting bill died:', ', '.join(map(lambda i: featurelist[i], wsorted[-30:][::-1]))

def logreg_gridsearch(trainX, trainY, devX, devy, maxiter_values, eta_values, alpha_values, batch_size_values):
    hyperparamdict = {}
    for batch_size in batch_size_values:
        for maxiter in maxiter_values:
            for eta in eta_values:
                for alpha in alpha_values:
                    w, b, train_accuracy = logreg_train(trainX, trainY, maxiter=maxiter, eta=eta, alpha=alpha, batch_size=batch_size)
                    print 'maxiter=', maxiter, 'eta=', eta, 'alpha=', alpha, 'batch_size=', batch_size, 'training accuracy=', train_accuracy
                    predictions = predict_confidence(w, b, devX)
                    hyperparamdict[(maxiter, eta, alpha, batch_size)] = {}
                    hyperparamdict[(maxiter, eta, alpha, batch_size)]['dev accuracy'] = get_meansq_accuracy(devy, predictions)
                    hyperparamdict[(maxiter, eta, alpha, batch_size)]['hyperplane'] = w, b
    return hyperparamdict

def main(dataloc):
    X, fullY = load(dataloc)
    X = normalize(handle_missing(X))
    split_sets(X, fullY)
    trainX, trainY = load("data/training.csv") # may want to un-hard-code later
    testX, testY = load("data/testing.csv") # may want to un-hard-code later

    #featurelist = map(lambda x:x[0], sorted(featuremap.items(), key=lambda x:x[1])) # from index to feature

if __name__=='__main__':
    """Run the main function"""
    if len(sys.argv)!=2:
        print 'Usage:', sys.argv[0], '{p,l}'
    else:
        main(sys.argv[1])
