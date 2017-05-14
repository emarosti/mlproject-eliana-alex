from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import sys
from data_loading import *

def train(trainX, trainY, k):
    """Train the knn classifier on the data
    k = number of clusters
    """
    accuracies = []

    for i in range(trainY.shape[1]-1):
        nearestneibs = KNeighborsClassifier(n_neighbors=k)
        nearestneibs.fit(trainX,trainY[:,i])
        accuracies.append(nearestneibs.score(trainX,trainY[:,i]))

    return np.array(accuracies), nearestneibs

def predict(testX, testY, nearestneibs):
    """ Make predictions on the test data
    """
    preds = []
    for i in range(testY.shape[1]-1):
        preds.append(nearestneibs.score(testX,testY[:,i]))
    return np.array(preds)


def main(dataloc):
    """
    MAIN METHOD
    """

    features = load_features("data/protein_features.csv")

    # currently hard-coded, should be argument:
    n_splits = 5
    n_chains = 10
    print "Using %i splits and %i chains" % (n_splits, n_chains)
    features = load_features(dataloc+"/protein_features.csv")
    X, fullY = load(dataloc+"/data.csv")
    split_sets(X, fullY, splits=n_splits)

    imp_sum = np.zeros((fullY.shape[1]-1, X.shape[1])) # initialize array to calculate weight averages
    mean_acc = np.zeros((n_splits*n_chains, fullY.shape[1]-1))
    test_acc = np.zeros((n_splits*n_chains, fullY.shape[1]-1))
    for i in range(n_splits): # or could be passed as an argument
        trainX, trainY = load((dataloc+"/train_"+str(i+1)+".csv"))
        devX, devY = load((dataloc+"/dev_"+str(i+1)+".csv"))
        testX, testY = load((dataloc+"/test_"+str(i+1)+".csv"))
        X = np.concatenate((trainX, devX, testX), axis=0)
        indicesX = [trainX.shape[0], devX.shape[0], testX.shape[0]]
        for j in range(n_chains):
            tmpX = missing_rnorm(X, X)
            tmpX = standardize(tmpX, X)
            tmptrainX = tmpX[:indicesX[0],:]

            accuracies, nearestneibs = train(trainX, trainY, 10)
            testaccuracies = predict(testX, testY, nearestneibs)
            #imp_sum += importances
            mean_acc[((i*n_chains)+j),:] = accuracies
            test_acc[((i*n_chains)+j),:] = testaccuracies


    #print "shape of summary:", imp_sum.shape, mean_acc.shape
    #print "shape per item:", accuracies.shape, importances.shape
    print "mean accuracies:", np.mean(mean_acc, axis=0)
    print "test accuracies:", np.mean(test_acc, axis=0)


if __name__=='__main__':
    if len(sys.argv)!=2:
        print 'Usage: python random_forest.py dataloc'
    else:
        main(sys.argv[1])
