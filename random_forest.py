from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys
from data_loading import load

def train(trainX, trainY, k, criterion, maxfeats):
    """
    k number of decision trees in the random forest
    """
    randoforest = RandomForestClassifier(n_estimators=k, criterion=criterion, max_features=maxfeats)
    randoforest.fit(trainX,trainY)
    randoforest.apply(trainX)
    return randoforest.score(trainX,trainY)

def main():
    """
    what do you call a sleepy trex? a dino-snore lol
    """

    trainX, trainY = load("data/train.csv") # may want to un-hard-code later
    devX, devY = load("data/dev.csv") # may want to un-hard-code later
    testX, testY = load("data/test.csv") # may want to un-hard-code later

    accuracy = train(trainX, trainY, 10, 'entropy', 70)

    print 'accuracy', accuracy

if __name__=='__main__':
    if len(sys.argv)!=1:
        print 'Usage: python linear_classifer.py'
    else:
        main()
