from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys
from data_loading import *

def train(trainX, trainY, k, criterion, maxfeats):
    """
    k number of decision trees in the random forest
    """
    accuracies = []
    importances = []
    for i in range(trainY.shape[1]-1):
        randoforest = RandomForestClassifier(n_estimators=k, criterion=criterion, max_features=maxfeats, max_depth=2)
        randoforest.fit(trainX,trainY[:,i])
        accuracies.append(randoforest.score(trainX,trainY[:,i]))
        importances.append(randoforest.feature_importances_)
        # randoforest.apply(trainX)
    return np.array(accuracies), np.array(importances), randoforest

def predict(testX, testY, randoforest):
    preds = []
    for i in range(testY.shape[1]-1):
        preds.append(randoforest.score(testX,testY[:,i]))
    return np.array(preds)

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
        print "Best features for predicting label", i, all_names[i,:k]
        # negative/low weights are not informative
    return all_names


def main(dataloc):
    """
    what do you call a sleepy trex? a dino-snore lol
    """

    features = load_features("data/protein_features.csv")

    # currently hard-coded, should be argument:
    n_splits = 5
    n_chains = 10
    print "Using %i splits and %i chains" % (n_splits, n_chains)
    features = load_features(dataloc+"/protein_features.csv")
    X, fullY = load(dataloc+"/data.csv")
    split_sets(X, fullY, splits=n_splits)

    sum_import = np.zeros((fullY.shape[1]-1, X.shape[1])) # initialize array to calculate weight averages
    mean_acc = np.zeros((n_splits*n_chains, fullY.shape[1]-1))
    test_acc = np.zeros((n_splits*n_chains, fullY.shape[1]-1))
    for i in range(n_splits): # or could be passed as an argument
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

            accuracies, importances, randoforest = train(tmptrainX, trainY, 5, 'entropy', 70)
            testaccuracies = predict(tmptestX, tmptestY, randoforest)
            sum_import += importances
            mean_acc[((i*n_chains)+j),:] = accuracies
            test_acc[((i*n_chains)+j),:] = testaccuracies


    print "shape of summary:", sum_import.shape, mean_acc.shape
    print "shape per item:", accuracies.shape, importances.shape
    print "mean accuracies:", np.mean(mean_acc, axis=0)
    print "test accuracies:", np.mean(test_acc, axis=0)
    sum_import = sum_import/ (n_splits * n_chains)

    # accuracies, importances, randoforest = train(trainX, trainY, 10, 'entropy', 70)
    # #TO DO: reduce number of trees or prune trees to lower overfitting
    # testaccuracies = predict(testX, testY, randoforest)

    ordered_feat = extract_sig_features(6, features, sum_import, 'top')

if __name__=='__main__':
    if len(sys.argv)!=2:
        print 'Usage: python random_forest.py dataloc'
    else:
        main(sys.argv[1])
