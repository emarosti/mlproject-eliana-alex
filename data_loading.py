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
    X = data[:, 0:data.shape[1]-4] # hard-coded for number of class columns
    fullY = data[:, data.shape[1]-4:] # hard-coded for number of class columns
    return (X, fullY)

def split_sets(X, fullY, test_size=0.2):
    """given two matrices of data and associated labels,
    split the dataset into a training and testing set (of proportion test_size,
    default to 20%)
    """
    data = X
    labels = fullY[:,-1] # last column of 8 classes (0-7)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    for train_indices, test_indices in sss.split(data, labels): # split returns generator type
        split_files(train_indices, data, fullY, "data/training.csv") # hard-coded because n_split = 1
        split_files(test_indices, data, fullY, "data/testing.csv") # hard-coded because n_split = 1

def split_files(index_list, data, labels, outfile):
    """HELPER FUNCTION
    saves selected indices of provided data and labels into a .csv file
    """
    data_out = []
    data_csv = open(outfile, 'w')
    for i in index_list:
        entry = data[i,:]
        entry = np.append(entry, labels[i,:])
        data_out.append(entry)
    np.savetxt(outfile, data_out, fmt='%1.6f', delimiter=',', newline='\n') # labels are also printed as floats

def empty_vals():
    """
    """
    pass

def main(dataloc):
    X, fullY = load(dataloc)
    split_sets(X, fullY)

if __name__=='__main__':
    if len(sys.argv)!=2:
        print 'Usage: python data_loading.py csv-data'
    else:
        main(sys.argv[1])
