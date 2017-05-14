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
    X = data[:, 0:data.shape[1]-5] # hard-coded for number of class columns
    fullY = data[:, data.shape[1]-5:] # hard-coded for number of class columns
    return (X, fullY)

def split_sets(X, fullY, test_size=0.2, splits=1):
    """given two matrices of data and associated labels,
    split the dataset into a training, dev, and testing set (of proportion test_size,
    default to 20%, where testing and dev are each half the test_size), some number of
    times (splits) for future cross-validation
    """
    data = X
    labels = fullY[:,-1] # last column of 8 classes (0-7)
    sss = StratifiedShuffleSplit(n_splits=splits, test_size=test_size, random_state=0)
    i = 1
    for train_indices, testdev_indices in sss.split(data, labels):
        split_csv(train_indices, data, fullY, ("data/train_"+str(i)+".csv"))
        testdevX, testdevY = reshape(testdev_indices, data, fullY)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        for test_indices, dev_indices in sss.split(testdevX, testdevY):
            split_csv(dev_indices, testdevX, testdevY, ("data/dev_"+str(i)+".csv"))
            split_csv(test_indices, testdevX, testdevY, ("data/test_"+str(i)+".csv"))
        i += 1

def split_csv(index_list, data, labels, outfile):
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

def reshape(index_list, data, labels):
    """HELPER FUNCTION
    returns selected indices of provided data and labels as ndarrays in original data format
    """
    X = np.zeros((len(index_list), data.shape[1]))
    y = np.zeros((len(index_list), labels.shape[1]))
    for i in range(len(index_list)):
        X[i,:] = data[index_list[i],:]
        y[i,:] = labels[index_list[i],:]
    return X, y

def missing_mean(X, fullX):
    """ MISSING DATA
    replace missing values (0.0) with column/feature mean
    """
    stats = column_stats(fullX) # maximums and minimums are not accurate
    tmpX = np.copy(X)
    for col in range(X.shape[1]):
        for row in range(X.shape[0]):
            if tmpX[row,col] == 0.0: tmpX[row,col] = stats[2,col]
    return tmpX

def missing_rnorm(X, fullX):
    """ MISSING DATA
    replace missing values with random samples from a
    normal (Gaussian) distribution
    """
    stats = column_stats(fullX) # maximums and minimums are not accurate
    tmpX = np.copy(X)
    for col in range(X.shape[1]):
        for row in range(X.shape[0]):
            if tmpX[row,col] == 0.0: tmpX[row,col] = np.random.normal(stats[2,col], stats[3,col])
    return tmpX

def missing_rm(X, fullY, thresCol, thresRow):
    """ MISSING DATA
    remove columns/rows with percent of missing values
    less than or equal to respective thresholds
    **NOTE: must be called before splitting data into sets
    (cannot work with subset of X)
    **NOTE: deletes from columns and rows at end due to
    inability to modify fullX, therefore missing percentages
    are not recalculated after columns are first deleted
    """
    tmpX = np.copy(X)
    tmpY = np.copy(fullY)
    feat_nonzero = []
    for col in range(X.shape[1]):
        if (np.count_nonzero(X[:,col]) / float(X.shape[0])) <= thresCol:
            #print (np.count_nonzero(fullX[:,col], axis=0) / float(X.shape[0]))
            feat_nonzero.append(col)
    samp_nonzero = []
    for row in range(X.shape[0]):
        if (np.count_nonzero(X[row,:]) / float(X.shape[1])) <= thresRow:
            #print (np.count_nonzero(fullX[row,:], axis=1) / float(X.shape[1]))
            samp_nonzero.append(row)
    print "Deleting columns", feat_nonzero
    print "Deleting rows", samp_nonzero
    # flipped axis notation from count_nonzero
    tmpX = np.delete(np.delete(tmpX, feat_nonzero, axis=1), samp_nonzero, axis=0)
    tmpY = np.delete(tmpY, samp_nonzero, axis=0)
    print "X size from", X.shape, "to", tmpX.shape, "\nY size from", fullY.shape, "to", tmpY.shape
    return tmpX, tmpY


def normalize(X, fullX):
    """ FEATURE SCALING
    (x - min) / (max - min)
    """
    stats = column_stats(fullX)
    for col in range(X.shape[1]):
        max_min = stats[0,col] - stats[1,col]
        for row in range(X.shape[0]):
            X[row,col] = (X[row,col] - stats[1,col]) / max_min
    return X

def standardize(X, fullX):
    """ FEATURE SCALING
    (x - mean) / stdev
    """
    stats = column_stats(fullX)
    for col in range(X.shape[1]):
        stdev = np.std(X[:,col])
        for row in range(X.shape[0]):
            X[row,col] = (X[row,col] - stats[2,col]) / stdev
    return X

def column_stats(X):
    """HELPER FUNCTION
    returns 3 x D array with column maximum, minimum, mean, and stdev (w/o zero values)
    """
    column_stats = np.zeros((4,X.shape[1]))
    for col in range(X.shape[1]):
        zero_count = 0.0
        stdev = 0.0
        column_stats[0,col] = X[:,col].max() # calculate maximum
        column_stats[1,col] = X[:,col].min() # calculate minimum
        for row in range(X.shape[0]): # calculate mean
            if X[row,col] == 0: zero_count += 1
        column_stats[2,col] = X[:,col].sum() / (X.shape[0] - zero_count)
        for row in range(X.shape[0]): # calculate standard deviation
            if X[row,col] != 0: stdev += (X[row,col] - column_stats[2,col]) ** 2
        column_stats[3,col] = (stdev / (X.shape[0]-zero_count)) ** (0.5)
    return column_stats

def load_features(filename):
    """given a csv file of the feature names (proteins in this project),
    load the feature values into an array and return that array
    """
    data = np.loadtxt(filename, dtype='str', delimiter=',')
    return data

def main(dataloc):
    X, fullY = load((dataloc+"/data.csv")) # hard-coded
    X = missing_rm(X, fullY, .95, .90)
    #print 'X BEFORE:'
    #print X
    #X = missing_rnorm(X, X)
    #print 'X AFTER:'
    #print X
    #X = standardize(X, X)
    #split_sets(X, fullY, splits=5)
    pass


if __name__=='__main__':
    if len(sys.argv)!=2:
        print 'Usage: python data_loading.py csv-data'
    else:
        main(sys.argv[1])
