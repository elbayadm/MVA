#!/usr/bin/env python2
# -*- coding: utf-8 -*-


"""
Main script for link prediction in citations graph

Usage:
  link_prediction.py  [ -s <flag> ] [ -v <flag> ] [ -f <flag> ] [ -p <flag> ] [ -n <int> ]

Options:
  -h --help  Show the description of the program
  -s <flag> --subm <flag>  prepare submission file [default: True]
  -p <flag> --pre <flag>  load precomputed features [default: False]
  -v <flag> --val <flag>  split training set to validate [default: False]
  -f <flag> --file <flag>  save the features files [default: False]
  -n <int> --n_samples <int>  number of samples to train on [default: 10000]
"""

from __future__ import division
from docopt import docopt
from attribute_features import *
from semantic_features import *
from topological_features import *
from classifiers import *
import random
import csv
import numpy as np
import pandas as pd
import networkx as nx

if __name__ == "__main__":
	# Parse the arguments:
	arguments = docopt(__doc__)
	print(arguments)

	if arguments["--pre"]=="False":
		# Loading datasets
		# Load the training set
		Xfull = pd.read_csv("../data/training_set.txt", sep=" ", header=None, names=['source', 'target','link'])
		# Shuffling
		Xfull = Xfull.reindex(np.random.permutation(Xfull.index))
		Yfull = Xfull['link']
		Xfull.drop(['link'], axis = 1, inplace = True)
		print 'Full training set dimension: ',Xfull.shape

		if arguments["--val"]=="True":
			# Split training(reduced) and validation set:		
			n = int(arguments['--n_samples'])
			Xtrain = Xfull[:n]
			Xtrain.index = range(n)
			Ytrain = Yfull[:n]
			Ytrain.index = range(n)
			m = min(int(n + round(n/10)),Xfull.shape[0])
			Xval = Xfull[n+1:m]
			Yval = Yfull[n+1:m]

			print 'Train/validation subsets:'
			print "Train : ", Xtrain.shape , Ytrain.shape
			print "Validation : ", Xval.shape, Yval.shape
		else:
			Xtrain  = Xfull
			Ytrain  = Yfull	
	
		# Load node information
		nodes_info = pd.read_csv("../data/node_information.csv", header= None, 
		    names=["Id", "year", "title", "authors", "journal", "abstract"],
		    sep=",",index_col = "Id", encoding = 'utf-8')

		print 'Parsing the authors and their affiliations...'
		fix_auth_aff(nodes_info)

		# # Topological features:
		# 1. Dispersion
		# 2. Nb. common neighbors
		# 3. Jaccard coefficient
		# 4. Adamic/Adar coefficient
		# 5. Difference of inlinks
		# 6. Nb. of articles that cited the target
		# 7. Max cited author pagerank (authors citations graph)
		# 8. Are articles in the same community
		# 9. Are authors in the same community
		

		print 'Building the citations graph...'
		G = build_graph(Xtrain, Ytrain, nodes_info)
		print 'Building the authors graphs...'
		GAC = build_authors_citation_graph(Xtrain, Ytrain, nodes_info)
		GAA = build_authors_coauthorship_graph(nodes_info)

		if arguments["--file"]=="True":
			# Save graphs
			fh1 = open("../data/G_authors_citations.edgelist","wb")
			nx.write_edgelist(GAC,fh1)
			fh2 = open("../data/G_authors_coauthorship.edgelist","wb")
			nx.write_edgelist(GAA,fh2)
			fh3 = open("../data/G_articles.edgelist","wb")
			nx.write_edgelist(G,fh3)

		Xtrain = topologic_features(Xtrain, G, GAC, GAA, nodes_info)
		print 'Updated dimension: ',Xtrain.shape

		# Attribute features:
		# 1. Title overlap
		# 2. Difference in publication year
		# 3. Is self citation
		# 4. same journal
		# 5. Common authors
		# 6. Is same affiliation

		
		Xtrain = attribute_features(Xtrain, nodes_info)
		print 'Updated dimension: ', Xtrain.shape

		# # Semantic features:
		# 1. cosine similarity between the abstracts (tf-idf)
		# 2. cosine similarity between the abstracts (word2vec)

		Xtrain = semantic_features(Xtrain, nodes_info)
		print 'Updated dimension: ', Xtrain.shape
		if arguments["--file"]=="True":
			Xtrain.to_csv('../data/Xtrain.csv')
			Ytrain.to_csv('../data/Ytrain.csv')

		if arguments["--val"]=="True":
			print 'Same features for the validation set...'
			Xval = topologic_features(Xval, G, GAC, GAA, nodes_info,train=False)
			Xval = attribute_features(Xval, nodes_info, train=False)
			Xval = semantic_features(Xval, nodes_info, train=False)
		 	
		 	if arguments["--file"]=="True":
				Xval.to_csv('../data/Xval.csv')
				Yval.to_csv('../data/Yval.csv')

			print 'Training the SVM classifier'
			# SVM classification
			classifier, scaler = gridsearch_svm(Xtrain,Ytrain, Xval, Yval)
			classifier , scaler = train_rf(Xtrain, Ytrain, Xval, Yval)

		if arguments["--subm"]=="True":
			# Load the test set
			Xtest = pd.read_csv("../data/testing_set.txt", sep=" ", header=None, names=['source', 'target'])
			print 'Test set dimension: ', Xtest.shape
			print 'Same features for the test set...'
			Xtest = topologic_features(Xtest, G, GAC, GAA, nodes_info,train=False)
			Xtest = attribute_features(Xtest, nodes_info, train=False)
			Xtest = semantic_features(Xtest, nodes_info, train=False)
			Xtest.to_csv('../data/Xtest.csv')
			if 'classifier' in locals():
				print 'Predicting for the test set'
				#pred = predict_svm(Xtest, classifier , scaler)
				pred = predict_rf(Xtest, classifier , scaler)
				pred = zip(range(len(Xtest)), pred)

				with open("submission.csv","wb") as pred1:
				    csv_out = csv.writer(pred1)
				    csv_out.writerow(['id','category'])
				    for row in pred:
				        csv_out.writerow(row)
			else:
				classifier, scaler = train_rf(Xtrain, Ytrain)
				pred = predict_svm(Xtest, classifier , scaler)
				pred = zip(range(len(Xtest)), pred)

				with open("submission.csv","wb") as pred1:
				    csv_out = csv.writer(pred1)
				    csv_out.writerow(['id','category'])
				    for row in pred:
				        csv_out.writerow(row)


	else:
		Xtrain = pd.read_csv("../data/Xtrain.csv", sep=",", index_col=0, header=0)
		Ytrain = pd.read_csv("../data/Ytrain.csv", sep=",", index_col=0, header=None, names=['link'])
		Ytrain = Ytrain['link']
		print 'Training set: ', Xtrain.shape, Ytrain.shape

		Xval = pd.read_csv("../data/Xval.csv", sep=",", index_col=0, header=0)
		Yval = pd.read_csv("../data/Yval.csv", sep=",", index_col=0, header=None, names = ['link'])
		Yval = Yval['link']
		print 'Validation set: ', Xval.shape, Yval.shape
	
		print 'Training the classifier'
		# SVM classification
		#classifier, scaler = gridsearch_svm(Xtrain,Ytrain, Xval, Yval)
		#classifier, scaler = train_svm(Xtrain,Ytrain, Xval, Yval)
		#classifier , scaler = train_rf(Xtrain, Ytrain, Xval, Yval)
		#scaler = train_NN(Xtrain,Ytrain, Xval, Yval)
		classifier, scaler = train_logit(Xtrain,Ytrain, Xval, Yval)

		
		if arguments["--subm"]=="True":
			Xtest = pd.read_csv("../data/Xtest.csv", sep=",", index_col=0, header=0)
			predictions_SVM = predict_svm(Xtest, svm_model , scaler_svm)
			predictions_SVM = zip(range(len(Xtest)), predictions_SVM)
			with open("submission.csv","wb") as pred1:
			    csv_out = csv.writer(pred1)
			    csv_out.writerow(['id','category'])
			    for row in predictions_SVM:
			        csv_out.writerow(row)
			


