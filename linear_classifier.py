#from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
#http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
import numpy as np
import sys
import data_loading     # contains helper functions

# 0/1 NOTES
# control/trisomy; saline/memantine; SC/CS

def classifier(X, y, classi, alpha, eta0, epoch, learn):
    """
    type = 'log', 'hinge'
    learn = 'constant', 'optimal', 'invscaling'
    """
    weights = []
    intercepts = []
    for i in range(y.shape[1]-1): # currently ignoring last 8-class column
        sgd = SGDClassifier(loss=classi, alpha=alpha, eta0=eta0, n_iter=epoch,
            learning_rate=learn, verbose=0)
        sgd.fit(X, y[:,i])
        weights.append(sgd.coef_)
        intercepts.append(sgd.intercept_)
    return sgd, weights, intercepts

def svc(X, y):
    """
    """
    #weights = [] # may make into ndarray
    #intercepts = []
    for i in range(y.shape[1]-1):
        clf = SVC(kernel='rbf') # vs. default 'rbf' ..alter later?
        clf.fit(X, y[:,i])
        #weights.append(clf.coef_)
        #intercepts.append(clf.coef_)
    return clf#, weights, intercepts

def accuracy(clf, testX, testY):
    """
    for SVC
    """
    accuracies = []
    for i in range(testY.shape[1]-1):
        accuracies.append(clf.score(testX, testY[:,i]))
    return accuracies

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

def rebind(trainX, devX, testX, ):
    """HELPER FUNCTION
    """
    X = np.append(trainX, devX, testX)


    return retrainX, redevX, retestX

def main(dataloc, splits, chains): # testing 5 splits, 10 chains
    # ??? SHOULD ALREADY HAVE BEEN RUN FROM data_loading.py
    n_splits = int(splits)
    n_chains = int(chains)
    X, fullY = data_loading.load(dataloc+"/data.csv")
    data_loading.split_sets(X, fullY, splits=n_splits)

    mean_weights = np.zeros((n_splits*n_chains, X.shape[1])) # initialize array to calculate weight averages
    mean_acc = np.zeros((n_splits*n_chains, fullY.shape[1]-1))
    for i in range(n_splits):
        trainX, trainY = data_loading.load((dataloc+"/train_"+str(i+1)+".csv"))
        devX, devY = data_loading.load((dataloc+"/dev_"+str(i+1)+".csv"))
        testX, testY = data_loading.load((dataloc+"/test_"+str(i+1)+".csv"))
        X = np.concatenate((trainX, devX, testX), axis=0)
        indicesX = [trainX.shape[0], devX.shape[0], testX.shape[0]]
        for j in range(n_chains):
            tmpX = data_loading.missing_rnorm(X, X)
            tmpX = data_loading.standardize(tmpX, X)
            tmptrainX = tmpX[:indicesX[0],:]

            #clf, w, b = classifier(trainX, trainY, 'log', 0.0001, 0.0, 5, 'optimal')
            clf = svc(tmptrainX, trainY)
            accuracies = accuracy(clf, testX, testY)
            #print accuracies
            mean_acc[((i*n_chains)+j),:] = accuracies
            #mean_weights[((i*n_splits)+j),:] = ????
            
    print "mean accuracies:", np.mean(mean_acc, axis=0)
    #print "mean weights:", np.mean(mean_weights, axis=0)
"""
    hyperparamdict = gridsearch(trainX, trainY, devX, devY,
                                   classifier_types=['log', 'hinge'],
                                   maxiter_values=[5, 25], # number of epochs
                                   eta0_values=[0.1, 0.01], # eta0
                                   learn_values=['constant', 'optimal', 'invscaling'],
                                   alpha_values=[0.1, 0.00001]) # regularization weight
"""
    #print 'Hyperparameter search results:'
    #for params in hyperparamdict:
    #    print params, hyperparamdict[params]['dev accuracy']

    #best_hyperparams = max(hyperparamdict.items(), key=lambda x:x[1]['dev accuracy'])[0]


if __name__=='__main__':
    if (len(sys.argv) != 4):
        print 'Usage: python linear_classifer.py dataloc num_splits num_chains'
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
