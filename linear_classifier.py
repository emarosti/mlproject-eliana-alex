from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import numpy as np
import sys
from data_loading import *
#http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

# NOTE ON LABEL CONVENTIONS FOR 0/1
# control/trisomy; saline/memantine; SC/CS; no_learning/yes_learning

def classifier(X, y, classi, alpha, eta0, epoch, learn):
    """
    type = 'log', 'hinge' (does not handle if neither)
    learn = 'constant', 'optimal', 'invscaling'
    """
    weights = []
    for i in range(y.shape[1]-1): # currently ignoring last 8-class column
        sgd = SGDClassifier(loss=classi, alpha=alpha, eta0=eta0, n_iter=epoch,
             learning_rate=learn, verbose=0)
        sgd.fit(X, y[:,i])
        weights.append(sgd.coef_) # disregard the intercepts
    return sgd, weights # same as importances

def svc(X, y, kernel):
    """
    kernel = 'rbf', 'linear' (does not handle if neither)
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
    """
    accuracies = []
    for i in range(testY.shape[1]-1):
        accuracies.append(clf.score(testX, testY[:,i]))
    return accuracies

def extract_sig_features(k, features, importances, direction):
    """
    given an array of features names and an array of importance values
    for each feature, return the k most important features in order of
    decreasing magnitude for both positive/negative predictions
    """
    imps = np.copy(importances)
    all_names = []
    args = np.argsort(imps, axis=1)
    for i in range(args.shape[0]):
        feat_names = []
        for j in range(args.shape[1]):
            feat_names.append(features[args[i,j]])
        all_names.append(feat_names)
    all_names = np.array(all_names)
    for i in range(args.shape[0]):
        print "Best features for predicting 0 for label", i, all_names[i,:k]
        print "Best features for predicting 1 for label", i, all_names[i,-k:][::-1]
    return all_names

def main(dataloc, splits, chains): # testing 5 splits, 10 chains
    n_splits = int(splits)
    n_chains = int(chains)
    features = load_features(dataloc+"/protein_features.csv")
    X, fullY = load(dataloc+"/data.csv")
    X, fullY = missing_rm(X, fullY, .95, .90)
    split_sets(X, fullY, splits=n_splits)

    sum_acc = np.zeros((4, fullY.shape[1]-1)) # hard-coded for number of classifiers tested
    sum_import = np.zeros((fullY.shape[1]-1, X.shape[1])) # initialize array to calculate weight averages
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

            clf1, w1 = classifier(trainX, trainY, 'log', 0.0001, 0.0, 5, 'optimal')
            clf2, w2 = classifier(trainX, trainY, 'hinge', 0.0001, 0.0, 5, 'optimal')
            clf3 = svc(tmptrainX, trainY, 'rbf')
            clf4, importances = svc(tmptrainX, trainY, 'linear')

            sum_acc[0,:] += accuracy(clf1, tmptestX, tmptestY)
            sum_acc[1,:] += accuracy(clf2, tmptestX, tmptestY)
            sum_acc[2,:] += accuracy(clf3, tmptestX, tmptestY)
            sum_acc[3,:] += accuracy(clf4, tmptestX, tmptestY)
            sum_import += importances

    print "test accuracies, 'logistic'", (sum_acc[0,:] / (n_splits*n_chains))
    print "test accuracies, 'svm (hinge)'", (sum_acc[1,:] / (n_splits*n_chains))
    print "test accuracies, 'rbf':", (sum_acc[2,:] / (n_splits*n_chains))
    print "test accuracies, 'linear':", (sum_acc[3,:] / (n_splits*n_chains))

    ordered_feat = extract_sig_features(6, features, sum_import, 'top')

if __name__=='__main__':
    if (len(sys.argv) != 4):
        print 'Usage: python linear_classifer.py dataloc num_splits num_chains'
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
