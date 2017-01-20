#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
Semantic features :
1. cosine similarity between the abstracts (tf-idf)
2. cosine similarity between the abstracts (word2vec)
'''

import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import word2vec
import numpy as np
from time import time
import pyprind
import scipy.io as io
from utils import *

def text_words_cleanser(text, stemmer, merge = False, stpwd = False):
	'''
	Convert a text to a sequence of cleaned words
	caveat = too slow when merging short words
	'''
	# lower & tokenize
	words = text.lower().split()
	# merge words with a single character (~ mathematic formulas)
	if merge:
		words_=[]
		buff = ''
		for i in range(len(words)-1):
			word = words[i]
			if len(word)>1:
				words_.append(word)
			else:
				buff = buff+word
				if len(words[i+1])>1:
					words_.append(buff)
					buff = ''
		if buff =='':
			words_.append(words[-1])
		else:
			words_.append(buff+words[-1])
		words = words_
	# remove stopwords:
	if stpwd:
		words = [w for w in words if w not in stpwds and len(w)>1]
	# stemming:
	words = [stemmer.stem(w) for w in words]
	return words

def text_sentences(text, tokenizer, stemmer):
	'''
	Split text into sentences
	'''
	merge = True
	stpwd = True
	sentences_ = tokenizer.tokenize(text.strip())
	sentences = []
	for sentence in sentences_:
		if len(sentence) > 0:
			sentences.append(text_words_cleanser(sentence, stemmer, merge, stpwd))
	return sentences



def get_corpus(info):
	sentences = []
	print 'Gathering the training corpus'
	bar = pyprind.ProgBar(len(info['abstract']),bar_char='█', width=barwidth)
	for abstract in info['abstract']:
		sentences += text_sentences(abstract, tokenizer, stemmer)
		bar.update()
	np.save('../data/corpus', sentences)
	return sentences

#----------------------------------------------
#				    Tf-Idf
#----------------------------------------------

def abstracts_to_tfidf(info):
	#corpus = get_corpus(info)
	corpus = np.load('../data/corpus.npy')
	corpus = [(' ').join(sentence) for sentence in corpus]
	print 'Corpus collected'
	vectorizer = TfidfVectorizer()
	t0 = time()
	features = vectorizer.fit_transform(corpus)
	t_time = time() - t0
	print("Transformation time: %0.3fs" % t_time)
	print "Dimension of Tf-Idf features", features.shape[1]
	io.savemat('../data/TFIDF_abstracts',{'TFIDF': features})
	return features


#----------------------------------------------
#				    Word2Vec
#----------------------------------------------

def train_word2vec(info=[]):
	'''
	Gensim Word2vec's implementation:
	'''
	t0 = time()
	logging.basicConfig(format='%(message)s', level=logging.INFO)
	#sentences = get_corpus(info)
	sentences = np.load('../data/corpus.npy')
	model = word2vec.Word2Vec( sentences,\
        workers   = 4,\
        size      = num_features_abstracts,\
        min_count = 20,\
        window    = 15,\
        sample    = 1e-3
    )
    
    # save the model for later use. You can load it later using Word2Vec.load()
   	fname = "../data/word2vec_"+str(num_features_abstracts)+"_20_15"
   	model.save(fname)

   	# If training finished (=no more updates, only querying)
   	# init_sims  trim unneeded model memory 
   	model.init_sims(replace=True)
   	w_time = time()- t0
   	print("Word2vec runtime: %0.3fs" % w_time)
   	return model


def make_word2vec(words_list, model, num_features):
    '''
    Average the word vectors in a list of words
    '''
    F = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    # the model's vocabulary
    index2word = set(model.index2word)
    # if word in words_list add to the features
    representor = []
    for word in words_list:
        if word in index2word: 
            nwords += 1
            F += model[word]
    #take the average
    if nwords != 0:
    	F = F/nwords  	
    return F

def make_abstracts_word2vec(model, num_features, corpus, info):
    '''
    The average feature vector (make_word2vec) for each abstract
    '''
    bar = pyprind.ProgBar(len(corpus), bar_char='█', width=barwidth)
    abstract_word2vec = []
    for abstract in corpus:
        abstract_word2vec.append(make_word2vec(abstract, model, num_features))
        bar.update()
    info['abstract word2vec'] = abstract_word2vec


def semantic_features(X, info, train = True):
	print 'Semantic features...'
	F = X.copy()
	num_features = num_features_abstracts
	cosine_abstracts = []
	cosine_tfidf_abstracts = []
	cosine_titles    = []
	if train:
		print 'Training the word2vec model...'
		w2vmodel = train_word2vec(info)
		print "Computing the word2vec features for each abstract"
		corpus = np.load('../data/corpus.npy')
		make_abstracts_word2vec(w2vmodel, num_features, corpus, info)
		# make_titles_word2vec(w2vmodel, num_features, info)
		print 'Computing the tf-ifd of the abstracts'
		TFIDF = abstracts_to_tfidf(info)
	else:
		TFIDF = io.loadmat('../data/TFIDF_abstracts')
		TFIDF = TFIDF['TFIDF']
		print TFIDF.shape
		w2vmodel = word2vec.Word2Vec.load("../data/word2vec_"+str(num_features_abstracts)+"_20_15")
	print 'Computing the semantic features...'
	indices = {k: v for(k, v) in [(info.index[i], i) for i in range(0, len(info.index))]}
	bar = pyprind.ProgBar(F.shape[0], bar_char='█', width=barwidth)
	for idx, edge in F.iterrows():
		ids = int(edge[0])
		idt = int(edge[1])
		source_info = info.loc[ids]
		target_info = info.loc[idt]
		# Word2Vec
		source_abstract = source_info['abstract word2vec']
		target_abstract = target_info['abstract word2vec']
		cosine_abstracts.append(cosine_similarity(source_abstract.reshape(1, -1),target_abstract.reshape(1, -1)))
		# TF-IDF
		cosine_tfidf_abstracts.append(cosine_similarity(TFIDF[indices[ids]],TFIDF[indices[idt]]))
		bar.update()
	# Build the set features array
	F['abstract tfidf cosine'] = np.squeeze(cosine_tfidf_abstracts)
	F['abstract word2vec cosine'] = np.squeeze(cosine_abstracts)
	return F

