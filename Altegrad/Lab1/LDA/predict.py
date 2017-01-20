import numpy as np
import scipy as sp
import scipy.linalg as linalg


def predict(X, projected_centroid, W):
    
    """Apply the trained LDA classifier on the test data 
    X: test data
    projected_centroid: centroid vectors of each class projected to the new space
    W: projection matrix computed by LDA
    """


    # Project test data onto the LDA space defined by W
    projected_data  = np.dot(X, W)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in the code to implement the classification
    # part of LDA. Follow the steps given in the assigment.
    
    m = X.shape[0]
    K = projected_centroid.shape[0]
    label = np.zeros((m))

    for i in range(0,m):
        dist = float('inf');
        for j in range(0,K):
            d= np.sum((projected_data[i,:] - projected_centroid[j])**2)
            if d<dist:
                dist=d
                label[i]=j
    
    
    # =============================================================

    # Return the predicted labels of the test data X
    return label