from sklearn.linear_model import SGDClassifier
import numpy as np
import sys
from data_loading import load

def classifier(X, y, classi, alpha, eta0, epoch, learn):
    """
    type = 'log', 'hinge'
    learn = 'constant', 'optimal', 'invscaling'
    """
    weights = []
    for i in range(y.shape[1]-1): # currently ignoring last 8-class column
        sgd = SGDClassifier(loss=classi, alpha=alpha, eta0=eta0, n_iter=epoch,
            learning_rate=learn, verbose=0)
        sgd.fit(X, y[:,i])
        weights.append(sgd.coef_)
    return sgd, weights

def accuracy(sgd, testX, testY):
    """
    """
    accuracies = []
    for i in range(testY.shape[1]-1):
        accuracies.append(sgd.score(testX, testY[:,i]))
    return accuracies

def logreg_gridsearch():
    pass

def main():
    """SHOULD ALREADY HAVE BEEN RUN FROM data_loading.py
    X, fullY = load(dataloc)
    X = data_loading.missing_mean(X) # not imported
    X = data_loading.standardize(X) # not imported
    data_loading.split_sets(X, fullY) # not imported"""

    trainX, trainY = load("data/train.csv") # may want to un-hard-code later
    devX, devY = load("data/dev.csv") # may want to un-hard-code later
    testX, testY = load("data/test.csv") # may want to un-hard-code later

    sgd, weights = classifier(trainX, trainY, 'log', 0.0001, 0.0, 5, 'optimal')
    accuracies = accuracy(sgd, testX, testY)

    print "weights:", weights, "\naccuracies:", accuracies

if __name__=='__main__':
    if len(sys.argv)!=1:
        print 'Usage: python linear_classifer.py'
    else:
        main()
