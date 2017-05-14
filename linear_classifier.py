#from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
#http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
import numpy as np
import sys
from data_loading import *  # contains helper functions

# 0/1 NOTES
# control/trisomy; saline/memantine; SC/CS; no_learning/yes_learning

def svc(X, y, kernel):
    """
    kernel = 'rbf', 'linear'
    """
    importances = np.zeros((y.shape[1]-1, X.shape[1]))
    for i in range(y.shape[1]-1):
        clf = SVC(kernel=kernel) # could change to linear
        clf.fit(X, y[:,i])
        if kernel == 'linear': importances[i,:] = clf.coef_
    if kernel == 'rbf': return clf
    if kernel == 'linear': return clf, importances

def accuracy(clf, testX, testY):
    """
    for SVC
    """
    accuracies = []
    for i in range(testY.shape[1]-1):
        accuracies.append(clf.score(testX, testY[:,i]))
    return accuracies

def extract_sig_features(k, features, importances, direction):
    """
    given an array of features names and an array of importance values
    for each feature, return the k most important features in order of
    decreasing magnitude
    direction = 'top', 'bottom'
    """
    imps = np.copy(importances)
    feat_names = []
    if direction == 'top': a = np.argsort(imps)[-k:][::-1]
    if direction == 'bottom': a = np.argsort(imps)[:k]
    for i in range(len(a)):
        feat_names.append(features[a[i]])
    return np.array(feat_names)

def main(dataloc, splits, chains): # testing 5 splits, 10 chains
    n_splits = int(splits)
    n_chains = int(chains)
    features = load_features(dataloc+"/protein_features.csv")
    X, fullY = load(dataloc+"/data.csv")
    print X.shape, fullY.shape
    X, fullY = missing_rm(X, fullY, .95, .90)
    split_sets(X, fullY, splits=n_splits)

    sum_import = np.zeros((fullY.shape[1]-1, X.shape[1])) # initialize array to calculate weight averages
    mean_acc1 = np.zeros((n_splits*n_chains, fullY.shape[1]-1))
    mean_acc2 = np.zeros((n_splits*n_chains, fullY.shape[1]-1))
    for i in range(n_splits):
        trainX, trainY = load((dataloc+"/train_"+str(i+1)+".csv"))
        devX, devY = load((dataloc+"/dev_"+str(i+1)+".csv"))
        testX, testY = load((dataloc+"/test_"+str(i+1)+".csv"))
        X = np.concatenate((trainX, devX, testX), axis=0)
        indicesX = [trainX.shape[0], devX.shape[0], testX.shape[0]]
        for j in range(n_chains):
            tmpX = standardize(missing_rnorm(X, X), X)
            tmptrainX = tmpX[:indicesX[0],:]
            tmptestX = tmpX[indicesX[0]:,:] # use both dev+test splits as testing
            tmptestY = np.concatenate((devY, testY), axis=0)

            #clf, w, b = classifier(trainX, trainY, 'log', 0.0001, 0.0, 5, 'optimal')
            clf1 = svc(tmptrainX, trainY, 'rbf')
            clf2, importances = svc(tmptrainX, trainY, 'linear')
            accuracies1 = accuracy(clf1, tmptestX, tmptestY)
            accuracies2 = accuracy(clf2, tmptestX, tmptestY)
            mean_acc1[((i*n_chains)+j),:] = accuracies1
            mean_acc2[((i*n_chains)+j),:] = accuracies2
            sum_import += importances

    print "test accuracies, 'rbf':", np.mean(mean_acc1, axis=0)
    print "test accuracies, 'linear':", np.mean(mean_acc1, axis=0)

    pos_sigfeat = []
    neg_sigfeat = []
    abs_sigfeat = []
    for i in range(importances.shape[0]):
        pos_sigfeat.append(extract_sig_features(6, features, sum_import[i,:], 'top'))
        neg_sigfeat.append(extract_sig_features(6, features, sum_import[i,:], 'bottom'))
        abs_sigfeat.append(extract_sig_features(6, features, abs(sum_import[i,:]), 'top'))
    pos_sigfeat = np.array(pos_sigfeat)
    neg_sigfeat = np.array(neg_sigfeat)
    abs_sigfeat = np.array(abs_sigfeat)


    print "most significant features -> 1, 'linear':\n", pos_sigfeat
    print "most significant features -> 0, 'linear':\n", neg_sigfeat
    print "most significant features, absolute value, 'linear':\n", abs_sigfeat

if __name__=='__main__':
    if (len(sys.argv) != 4):
        print 'Usage: python linear_classifer.py dataloc num_splits num_chains'
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
