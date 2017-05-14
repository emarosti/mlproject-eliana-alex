from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys
from data_loading import *

def train(trainX, trainY, k, criterion, maxfeats, maxdepth=None):
    """ Train the Random Forest classifier on the data
    k = number of decision trees in the random forest
    criterion = entropy
    maxfeats = maximum number of features to consider
    maxdepth = maximum depth of each decision tree within the forest
    """
    accuracies = []
    importances = []
    for i in range(trainY.shape[1]-1):
        randoforest = RandomForestClassifier(n_estimators=k, criterion=criterion, max_features=maxfeats, max_depth=maxdepth)
        randoforest.fit(trainX,trainY[:,i])

        accuracies.append(randoforest.score(trainX,trainY[:,i])) #accuracies
        importances.append(randoforest.feature_importances_) #importance values for features

    return np.array(accuracies), np.array(importances), randoforest

def predict(testX, testY, randoforest):
    """ Make predictions on the test data
    """
    preds = []
    for i in range(testY.shape[1]-1):
        preds.append(randoforest.score(testX,testY[:,i]))
    return np.array(preds)

def extract_sig_features(k, features, importances):
    """Given an array of features names and an array of importance values
    for each feature, return the k most important features
    """
    imps = np.copy(importances)
    imp_feats = []
    for i in range(k):
        a = np.argmax(imps)
        imp_feats.append(features[a])
        imps[a] = 0

    return np.array(imp_feats)



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
    X, fullY = missing_rm(X,fullY, .95, .9)
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

            accuracies, importances, randoforest = train(trainX, trainY, 10, 'entropy', trainX.shape[1])
            testaccuracies = predict(testX, testY, randoforest)
            imp_sum += importances
            mean_acc[((i*n_chains)+j),:] = accuracies
            test_acc[((i*n_chains)+j),:] = testaccuracies


    print "shape of summary:", imp_sum.shape, mean_acc.shape
    print "shape per item:", accuracies.shape, importances.shape
    print "mean accuracies:", np.mean(mean_acc, axis=0)
    print "test accuracies:", np.mean(test_acc, axis=0)
    imp_sum = imp_sum / (n_splits * n_chains)

    # accuracies, importances, randoforest = train(trainX, trainY, 10, 'entropy', 70)
    # #TO DO: reduce number of trees or prune trees to lower overfitting
    # testaccuracies = predict(testX, testY, randoforest)

    # print 'accuracies\n', accuracies
    # print 'importances\n', importances
    # print 'test accuracies\n', testaccuracies

    significant_features = []
    for i in range(importances.shape[0]):
        significant_features.append(extract_sig_features(12, features, imp_sum[i,:]))
    significant_features = np.array(significant_features)

    print 'Most significant features:\n', significant_features

if __name__=='__main__':
    if len(sys.argv)!=2:
        print 'Usage: python random_forest.py dataloc'
    else:
        main(sys.argv[1])
