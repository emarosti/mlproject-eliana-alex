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


def extract_sig_features(k, features, importances, direction):
    """
    Given an array of features names and an array of importance values
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


def main(dataloc, splits, chains, feats):
    """
    MAIN METHOD
    """
    n_splits = int(splits)
    n_chains = int(chains)
    n_feats = int(feats)
    print "Using %i splits and %i chains..." % (n_splits, n_chains)

    features = load_features(dataloc+"/protein_features.csv")
    X, fullY = load(dataloc+"/data.csv")
    X, fullY = missing_rm(X, fullY, .95, .90)
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

            accuracies, importances, randoforest = train(trainX, trainY, 10, 'entropy', trainX.shape[1])
            testaccuracies = predict(testX, testY, randoforest)
            imp_sum += importances

            mean_acc[((i*n_chains)+j),:] = accuracies
            test_acc[((i*n_chains)+j),:] = testaccuracies

    print "mean accuracies:", np.mean(mean_acc, axis=0)
    print "test accuracies:", np.mean(test_acc, axis=0)
    sum_import = sum_import/ (n_splits * n_chains)

    # accuracies, importances, randoforest = train(trainX, trainY, 10, 'entropy', 70)
    # #TO DO: reduce number of trees or prune trees to lower overfitting
    # testaccuracies = predict(testX, testY, randoforest)

    ordered_feat = extract_sig_features(n_feats, features, sum_import, 'top')

if __name__=='__main__':
    if len(sys.argv)!= 5:
        print 'Usage: python random_forest.py dataloc num_splits num_chains num_feats'
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
