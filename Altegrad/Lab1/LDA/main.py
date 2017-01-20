import math
import numpy as np
import scipy as sp
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from my_LDA import my_LDA
from predict import predict

# Load data (Wine dataset)
my_data = np.genfromtxt('wine_data.csv', delimiter=',')

acc = np.zeros((20))
for run in range(0,20): 
	# Split the data:
	np.random.shuffle(my_data) # shuffle datataset
	trainingData = my_data[:100,1:] # training data
	trainingLabels = my_data[:100,0] # class labels of training data
	testData = my_data[101:,1:] # training data
	testLabels = my_data[101:,0] # class labels of training data

	# Training LDA classifier
	W, projected_centroid, X_lda = my_LDA(trainingData, trainingLabels)

	# Perform predictions for the test data
	predictedLabels = predict(testData, projected_centroid, W)
	predictedLabels = predictedLabels+1


	# Compute accuracy
	counter = 0
	for i in range(predictedLabels.size):
	    if predictedLabels[i] == testLabels[i]:
	        counter += 1
	print 'Accuracy of LDA: %f' % (counter / float(predictedLabels.size) * 100.0)
	acc[run]=counter / float(predictedLabels.size) * 100.0
	W=None
	projected_centroid=None
	X_lda=None
	predictedLabels=None
print 'Average accuracy: ', np.mean(acc)
print 'Deviation: ', np.std(acc)
print 'Min: ', np.min(acc)
print 'Max: ',np.max(acc)
