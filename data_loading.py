import sys
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def load(filename):
    """given a CSV file where each row is a data point,
    with the last four columns being the labels and the rest being the vector,
    return a tuple consisting of two elements:
    (1) a matrix where each row is a vector, in the same order as they appear in the file
    (2) a matrix where the ith element has four labels describing the ith entry of the vector above.
    """
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 0:data.shape[1]-4]
    fullY = data[:, data.shape[1]-4:]
    return (X, fullY)

def split_sets():
    pass


def main(dataloc):
    X, fullY = load(dataloc)
    print "X:", X[0,:]
    print "Y:", fullY[0,:]

if __name__=='__main__':
    """Uncomment when ready"""
    if len(sys.argv)!=2:
        print 'Usage: python data_loading.py csv-data'
    else:
        main(sys.argv[1])
