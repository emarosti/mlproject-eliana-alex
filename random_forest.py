from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys
from data_loading import load, load_features

def train(trainX, trainY, k, criterion, maxfeats):
    """
    k number of decision trees in the random forest
    """
    accuracies = []
    importances = []
    for i in range(trainY.shape[1]):
        randoforest = RandomForestClassifier(n_estimators=k, criterion=criterion, max_features=maxfeats)
        randoforest.fit(trainX,trainY[:,i])
        accuracies.append(randoforest.score(trainX,trainY[:,i]))
        importances.append(randoforest.feature_importances_)
        # randoforest.apply(trainX)
    return np.array(accuracies), np.array(importances)

def predict(testX, testY):
    pass
    #for i in

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



def main():
    """
    what do you call a sleepy trex? a dino-snore lol
    """
    features = load_features("data/protein_features.csv")
    trainX, trainY = load("data/train.csv") # may want to un-hard-code later
    devX, devY = load("data/dev.csv") # may want to un-hard-code later
    testX, testY = load("data/test.csv") # may want to un-hard-code later

    accuracies, importances = train(trainX, trainY, 10, 'entropy', 70)
    #TO DO: reduce number of trees or prune trees to lower overfitting

    #print 'accuracies\n', accuracies
    #print 'importances\n', importances

    significant_features = []
    for i in range(importances.shape[0]):
        significant_features.append(extract_sig_features(12, features, importances[i,:]))
    significant_features = np.array(significant_features)

    print 'Most significant features:\n', significant_features

if __name__=='__main__':
    if len(sys.argv)!=1:
        print 'Usage: python linear_classifer.py'
    else:
        main()
















