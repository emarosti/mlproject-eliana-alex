from sklearn.linear_model import SGDClassifier
import numpy as np
import sys
from data_loading import load

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

def accuracy(sgd, testX, testY):
    """
    """
    accuracies = []
    for i in range(testY.shape[1]-1):
        accuracies.append(sgd.score(testX, testY[:,i]))
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

def main():
    """SHOULD ALREADY HAVE BEEN RUN FROM data_loading.py
    X, fullY = load(dataloc)
    X = data_loading.missing_mean(X) # not imported
    X = data_loading.standardize(X) # not imported
    data_loading.split_sets(X, fullY) # not imported"""

    trainX, trainY = load("data/train.csv") # may want to un-hard-code later
    devX, devY = load("data/dev.csv") # may want to un-hard-code later
    testX, testY = load("data/test.csv") # may want to un-hard-code later

    sgd, w, b = classifier(trainX, trainY, 'log', 0.0001, 0.0, 5, 'optimal')
    accuracies = accuracy(sgd, testX, testY)

    print "weights:", w, "\naccuracies:", accuracies

    hyperparamdict = gridsearch(trainX, trainY, devX, devY,
                                   classifier_types=['log', 'hinge'],
                                   maxiter_values=[5, 25], # number of epochs
                                   eta0_values=[0.1, 0.01], # eta0
                                   learn_values=['constant', 'optimal', 'invscaling'],
                                   alpha_values=[0.1, 0.00001]) # regularization weight

    print 'Hyperparameter search results:'
    for params in hyperparamdict:
        print params, hyperparamdict[params]['dev accuracy']

    best_hyperparams = max(hyperparamdict.items(), key=lambda x:x[1]['dev accuracy'])[0]


if __name__=='__main__':
    if len(sys.argv)!=1:
        print 'Usage: python linear_classifer.py'
    else:
        main()
