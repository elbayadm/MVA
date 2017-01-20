import numpy as np
import scipy as sp
import scipy.linalg as linalg

def my_LDA(X, Y):
    """
    Train a LDA classifier from the training set
    X: training data
    Y: class labels of training data

    """    
    
    classLabels = np.unique(Y) # different class labels on the dataset
    classNum = len(classLabels) # number of classes
    datanum, dim = X.shape # dimensions of the dataset
    totalMean = np.mean(X,0) # total mean of the data

    # ====================== YOUR CODE HERE ======================
    # Instructions: Implement the LDA technique, following the
    # steps given in the pseudocode on the assignment.
    # The function should return the projection matrix W,
    # the centroid vector for each class projected to the new
    # space defined by W and the projected data X_lda.
    
    classes = {classLabels[i]: X[Y==classLabels[i]] for i in range(classNum) };
    # -------------- Centroids:
    centroids = { i: np.mean(classes[ i ],axis=0) for i in classes.keys() };
    # -------------- Within class scatter matrix
    centered = {i:classes[ i ]-centroids[i]  for i in classes.keys() };
    Sw = sum ([ sum([np.dot(centered[ i ][j,:][np.newaxis,:].T , centered[ i ][j,:][np.newaxis,:] ) for j in range(len(centered[i])) ] )  for i in classes.keys() ]);
    
    #--------------- Between class scatter matrix
    Sb = sum( [ len(classes[ i ]) * np.dot( (centroids[ i ] - totalMean)[np.newaxis,:].T , (centroids[ i ] - totalMean)[np.newaxis,:] ) for i in classes.keys() ]);    
    
    #--------------- Computing W
    S,U = np.linalg.eig( np.dot( np.linalg.inv(Sw) , Sb ) );
    perm = np.argsort(S)[::-1];
    W = U[:,perm[:classNum-1]];

    #--------------- Projection onto the new space
    X_lda = np.dot( X , W );
    projected_centroid = np.dot( np.array( [ centroids[i] for i in centroids.keys() ] ) , W );

    # =============================================================

    return W, projected_centroid, X_lda