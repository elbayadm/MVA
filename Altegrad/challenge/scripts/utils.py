#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''

Parameters used for different features at different stages

'''
import nltk

def flatten_array(arr):
    arr_ = []
    for x in arr:
        if isinstance(x,list):
            for item in x:
                arr_.append(item)
        else:
            arr_.append(x)
    return set(arr_)

countries = set(nltk.corpus.gazetteers.words('countries.txt'))
countries = [x.lower() for x in countries]

stpwds = set(nltk.corpus.stopwords.words(["english", "spanish", "french", "italian"]))
stpwds = [x.lower() for x in stpwds] 
univ = ['university','u.','u','univ','univ.', 'universidade','universita','universitaet', 'Universitat', 'Universitet']
dept = ['department', 'dept.','dept','dipartimento']
inst = ['inst.','institute','inst','instituto']

stemmer = nltk.stem.PorterStemmer()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

num_features_affil  = 100
num_features_abstracts = 200
barwidth = 40