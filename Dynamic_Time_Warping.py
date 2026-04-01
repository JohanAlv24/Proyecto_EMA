from dtw import dtw
import numpy as np

def dtw_distance(x, y, normalize=False):
    alignment = dtw(x, y)
    return alignment.normalizedDistance if normalize else alignment.distance

def compute_dtw_matrix(X, normalize=False):
    n = len(X)
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            d = dtw_distance(X[i], X[j], normalize)
            D[i, j] = d
            D[j, i] = d 
            
    return D

def compute_dtw_test_matrix(X_test, X_train, normalize=False):
    n_test = len(X_test)
    n_train = len(X_train)
    
    D = np.zeros((n_test, n_train))
    
    for i in range(n_test):
        for j in range(n_train):
            D[i, j] = dtw_distance(X_test[i], X_train[j], normalize)
    
    return D