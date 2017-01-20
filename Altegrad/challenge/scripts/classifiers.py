#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
Classifiers
'''
from sklearn import svm, grid_search
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import sys
from time import time
from sklearn.utils.extmath import density
from sklearn.decomposition import PCA, FactorAnalysis

todrop = ['source','target'];

def drop_columns(X):
	X_ = X.drop(todrop, axis = 1)
	return X_

def scale_data(Xtrain, scaler=None):	
	X1 = drop_columns(Xtrain)
	if scaler is not None:
		X1 = pd.DataFrame(scaler.transform(X1), columns=X1.columns)
		return X1	
	else:
		scaler = MinMaxScaler(feature_range=(0.0, 1.0))
		X1 = pd.DataFrame(scaler.fit_transform(X1), columns=X1.columns)
		return X1, scaler

def train_svm(Xtrain,Ytrain, Xval=None, Yval=None):
	X1, scaler = scale_data(Xtrain)
	if Xval is not None:
		X2  = scale_data(Xval,scaler)
	# initialize basic SVM
	classifier = svm.SVC(verbose = True, shrinking=False, C=10, kernel='rbf')
	# train
	t0 = time()
	classifier.fit(X1, Ytrain)
	train_time = time() - t0
	print("train time: %0.3fs" % train_time)
	if Xval is not None:
		# prediction on validation set:
		t0 = time()
		pred = list(classifier.predict(X2))
		test_time = time() - t0
		print("test time:  %0.3fs" % test_time)
		if hasattr(classifier, 'coef_'):
			print("dimensionality: %d" % classifier.coef_.shape[1])
			print("density: %f" % density(classifier.coef_))
		print 'F1-score : ', f1_score(Yval, pred, average='binary')
		print("classification report:")
		print(classification_report(Yval, pred, target_names=['0','1'], digits=4))
		print("confusion matrix:")
		print(confusion_matrix(Yval, pred))
	return classifier, scaler

def gridsearch_svm(Xtrain,Ytrain, Xval, Yval):
	#---------------------------------- Scaling
	X1, scaler = scale_data(Xtrain)
	X2 = scale_data(Xval, scaler)
	#---------------------------------- Factor analysis
	fa = FactorAnalysis()
	X1 = fa.fit_transform(X1)
	X2 = fa.fit(X2)
	#---------------------------------- Cross validation and grid search
	cv = ShuffleSplit(len(Xtrain),n_iter=1, train_size=0.25, test_size = .03, random_state=0)
	params = {'C':[1, 10], 'kernel': ['rbf', 'linear']}
	svr = svm.SVC(verbose = True, shrinking=False)
	classifier = grid_search.GridSearchCV(svr, params,verbose=3, cv=cv)
	t0 = time()
	classifier.fit(X1, Ytrain)
	train_time = time() - t0
	print("train time: %0.3fs" % train_time)
	#---------------------------------- Prediction on validation set:
	t0 = time()
	pred = list(classifier.predict(X2))
	test_time = time() - t0
	print("test time:  %0.3fs" % test_time)
	if hasattr(classifier, 'coef_'):
		print("dimensionality: %d" % classifier.coef_.shape[1])
		print("density: %f" % density(classifier.coef_))
	print 'F1-score : ', f1_score(Yval,pred, average='binary')
	print("classification report:")
	print(classification_report(Yval, pred, target_names=['0','1'], digits=4))
	print("confusion matrix:")
	print(confusion_matrix(Yval, pred))
	return classifier, scaler

def predict_svm(X, model , scaler):
	X = scale_data(X,scaler)
	predictions_SVM = list(model.predict(X))
	return predictions_SVM


def train_NN(Xtrain, Ytrain, Xval, Yval):
	X1, scaler = scale_data(Xtrain)
	X2  = scale_data(Xval,scaler)
	sys.path.insert(1,'caffe_NN')
	from caffenet import citations_network, citations_predict
	t0 = time()
	citations_network(X1.as_matrix(),Ytrain.as_matrix(),X2.as_matrix(),Yval.as_matrix())
	train_time = time() - t0
	print("train time: %0.3fs" % train_time)
	t0 = time()
	pred = citations_predict(X2.as_matrix())
	test_time = time() - t0
	print("test time: %0.3fs" % test_time)
	print 'F1-score : ', f1_score(Yval, pred, average='binary')
	print("classification report:")
	print(classification_report(Yval, pred, target_names=['0','1'],digits=4))
	print("confusion matrix:")
	print(confusion_matrix(Yval, pred))
	return scaler

def predict_NN(X, scaler):
	sys.path.insert(1,'caffe_NN')
	from caffenet import citations_predict
	X_ = drop_columns(X, ['source','target'])
	X_ = pd.DataFrame(scaler.transform(X_), columns=X_.columns)
	predictions_NN = citations_predict(X_.as_matrix())
	return predictions_NN

def train_rf(Xtrain, Ytrain, Xval=None, Yval=None):
	X1, scaler = scale_data(Xtrain)
	classifier = RandomForestClassifier(n_estimators = 100, verbose=True)
	t0 = time()
	classifier.fit(X1, np.ravel(Ytrain))
	train_time = time() - t0
	print("train time: %0.3fs" % train_time)
	if Xval is not None:
		X2 = scale_data(Xval, scaler)
		t0 = time()
		pred = classifier.predict(X2.as_matrix())
		test_time = time() - t0
		print("test time: %0.3fs" % test_time)
		if hasattr(classifier, 'coef_'):
			print("dimensionality: %d" % classifier.coef_.shape[1])
			print("density: %f" % density(classifier.coef_))
		print 'F1-score : ', f1_score(Yval, pred, average='binary')
		print("classification report:")
		print(classification_report(Yval, pred, target_names=['0','1'],digits=4))
		print("confusion matrix:")
		print(confusion_matrix(Yval, pred))
	return classifier, scaler


def train_extraT(Xtrain, Ytrain):
	X1, scaler = scale_data(Xtrain)
	classifier = ExtraTreesClassifier(n_estimators=250,random_state=0)
	t0 = time()
	classifier.fit(X1, np.ravel(Ytrain))
	train_time = time() - t0
	print("train time: %0.3fs" % train_time)
	importances = classifier.feature_importances_
	std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
             axis=0)
	indices = np.argsort(importances)[::-1]
	results =  X1.loc[0]
	results[:] = importances
	print results
	return classifier, scaler

def predict_rf(X, classifier, scaler ):
	X_ = scale_data(X, scaler)
	predictions_RFC = classifier.predict(X_.as_matrix())
	return predictions_RFC


def recursive_feature_elimination(Xtrain,Ytrain):
	X1, scaler = scale_data(Xtrain)
	classifier = svm.SVC(kernel="linear", C=10)
	rfe = RFE(estimator=classifier, n_features_to_select=12, step=1)
	rfe.fit(X1, Ytrain)
	results = X1.loc[0]
	results[:] = rfe.ranking_
	print results
	return rfe

def train_logit(Xtrain, Ytrain, Xval=None, Yval=None):
	X1, scaler = scale_data(Xtrain)
	classifier = LogisticRegression(C=1, penalty='l1', tol=1e-4, verbose=True)
	t0 = time()
	classifier.fit(X1, Ytrain)
	train_time = time() - t0
	print("train time: %0.3fs" % train_time)
	if Xval is not None:
		X2 = scale_data(Xval, scaler)
		t0 = time()
		pred = classifier.predict(X2)
		test_time = time() - t0
		print("test time: %0.3fs" % test_time)
		if hasattr(classifier, 'coef_'):
			print("dimensionality: %d" % classifier.coef_.shape[1])
			print("density: %f" % density(classifier.coef_))
		print 'F1-score : ', f1_score(Yval, pred, average='binary')
		print("classification report:")
		print(classification_report(Yval, pred, target_names=['0','1'],digits=4))
		print("confusion matrix:")
		print(confusion_matrix(Yval, pred))
	return classifier, scaler
