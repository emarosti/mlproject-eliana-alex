#from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
#http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
import numpy as np
import sys
from data_loading import *  # contains helper functions

# 0/1 NOTES
# control/trisomy; saline/memantine; SC/CS; no_learning/yes_learning

# def classifier(X, y, classi, alpha, eta0, epoch, learn):
#     """ DEPRECATED
#     type = 'log', 'hinge'
#     learn = 'constant', 'optimal', 'invscaling'
#     """
#     weights = []
#     intercepts = []
#     for i in range(y.shape[1]-1): # currently ignoring last 8-class column
#         sgd = SGDClassifier(loss=classi, alpha=alpha, eta0=eta0, n_iter=epoch,
#             learning_rate=learn, verbose=0)
#         sgd.fit(X, y[:,i])
#         weights.append(sgd.coef_)
#         intercepts.append(sgd.intercept_)
#     return sgd, weights, intercepts

def svc(X, y):
    """
    """
    for i in range(y.shape[1]-1):
        clf = SVC(kernel='rbf') # could change to linear
        clf.fit(X, y[:,i])
    return clf

def accuracy(clf, testX, testY):
    """
    for SVC
    """
    accuracies = []
    for i in range(testY.shape[1]-1):
        accuracies.append(clf.score(testX, testY[:,i]))
    return accuracies

def extract_sig_features(k, features, importances):
    """given an array of features names and an array of importance values
    for each feature, return the k most important features
    """
    imps = np.copy(importances)
    imp_feats = []
    for i in range(k):
        a = np.argmax(imps)
        imp_feats.append(features[a])
        imps[a] = 0

    return np.array(imp_feats)

def show_significant_features(w, featurelist):
    wsorted = np.argsort(w)
    print 'Features predicting:', ', '.join(map(lambda i: featurelist[i], wsorted[:30]))
    print 'Features predicting bill died:', ', '.join(map(lambda i: featurelist[i], wsorted[-30:][::-1]))

def gridsearch(trainX, trainY, devX, devY, classifier_types, maxiter_values, eta0_values, learn_values, alpha_values):
    hyperparamdict = {}
    for classi in classifier_types:
        for maxiter in maxiter_values:
            for eta0 in eta0_values:
                for learn in learn_values:
                    for alpha in alpha_values:
                        sgd, w, b = classifier(trainX, trainY, classi=classi, epoch=maxiter, eta0=eta0, learn=learn, alpha=alpha)
                        print 'type=', classi, 'maxiter=', maxiter, 'eta0=', eta0, 'learn', learn, 'alpha=', alpha #NECESSARY? 'training accuracy=', train_accuracy
                        #predictions = sgd.predict(w, b, devX)
                        hyperparamdict[(classi, maxiter, eta0, learn, alpha)] = {}
                        hyperparamdict[(classi, maxiter, eta0, learn, alpha)]['dev accuracy'] = sgd.score(devX, devY)
                        hyperparamdict[(classi, maxiter, eta0, learn, alpha)]['hyperplane'] = w, b
    return hyperparamdict

def main(dataloc, splits, chains): # testing 5 splits, 10 chains
    n_splits = int(splits)
    n_chains = int(chains)
    X, fullY = load(dataloc+"/data.csv")
    X, fullY = missing_rm(X, fullY, .95, .90)
    split_sets(X, fullY, splits=n_splits)

    mean_weights = np.zeros((n_splits*n_chains, X.shape[1])) # initialize array to calculate weight averages
    mean_acc = np.zeros((n_splits*n_chains, fullY.shape[1]-1))
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
            clf = svc(tmptrainX, trainY)
            accuracies = accuracy(clf, tmptestX, tmptestY)
            mean_acc[((i*n_chains)+j),:] = accuracies

    print "mean accuracies:", np.mean(mean_acc, axis=0)

if __name__=='__main__':
    if (len(sys.argv) != 4):
        print 'Usage: python linear_classifer.py dataloc num_splits num_chains'
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
