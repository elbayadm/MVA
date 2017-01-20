#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''

Preporcess the nodes information and create attribute features :
1. Title overlap
2. Difference in publication year
3. Is self citation
4. same journal
5. Common authors
6. Is same affiliation
7. Is same affiliation (tfidf)

'''
import pyprind
import itertools
import numpy as np
import re
from gensim.models import word2vec
import logging
from semantic_features import make_word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from sklearn.metrics.pairwise import cosine_similarity
import scipy.io as io
from utils import *


def affiliations_to_tfidf(info):
    """
    Get Tf-Idf for the affiliations
    """    
    sentences = info['flat affiliation'].tolist()
    sentences = [(' ').join(s) if s!=['unknown'] else '' for s in sentences ]
    print 'Corpus collected'
    vectorizer = TfidfVectorizer()
    t0 = time()
    features = vectorizer.fit_transform(sentences)
    t_time = time() - t0
    print("Transformation time: %0.3fs" % t_time)
    print "Dimension of Tf-Idf features", features.shape[1]
    io.savemat('../data/TFIDF_affiliations',{'TFIDF': features})
    return features


def train_word2vec_affiliations(info):
    '''
    Gensim Word2vec's implementation:
    '''
    t0 = time()
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    sentences = info['flat affiliation'].tolist()
    sentences = [s.split(' ') for sublist in sentences for s in sublist if sublist!=['unknown']]
    model = word2vec.Word2Vec( sentences,\
        workers   = 4,\
        size      = num_features_affil,\
        min_count = 4,\
        window    = 3
    )
    
    # save the model for later use. You can load it later using Word2Vec.load()
    fname = "../data/affil_word2vec_"+str(num_features_affil)+"_4_3"
    model.save(fname)

    # If training finished (=no more updates, only querying)
    # init_sims  trim unneeded model memory 
    model.init_sims(replace=True)
    w_time = time()- t0
    print("Word2vec runtime: %0.3fs" % w_time)
    return model

def compare_affil(aff1, aff2, model, num_features):
    if (aff1==['unknown'] or aff2==['unknown']):
        return 0.0
    a1 = []
    for a in aff1:
        a1 += a.split(' ')
    a2 = []
    for a in aff2:
        a2 += a.split(' ')
    F1 = make_word2vec(a1, model,num_features)
    F2 = make_word2vec(a2, model,num_features)
    return np.squeeze(cosine_similarity(F1.reshape(1,-1),F2.reshape(1,-1)))

def parse_authors_affiliations(authors):
    if '((' in authors:
        # print 'Input: ', authors
        # format 1:
        # "R. A. Janik (1, 2), J. Wosiek (2) ((1) CEA-Saclay, (2) Jagellonian university cracow)"
        affiliations = authors[authors.find('((')+1:-1].split(',')
        aff_names = [format_affiliation_name(x[x.find(')')+1:]) for x in affiliations]
        auth = authors[:authors.find('((')].split(',')
        author_names = [x[:x.find('(')].strip() for x in auth]
        aff  = [x[x.find('(')+1:x.find(')')].strip() .split(' ') if '(' in x else ['unknown'] for x in auth]
        aff_names = [aff_names[int(i)-1] if i !='unknown' else 'unknown' for sublist in aff for i in sublist]
        # print 'Authors : ', author_names
        # print 'affiliations : ', aff_names
    elif '(' in authors:
        # print 'Input: ', authors
        # format 2:
        # j. t. lunardi, b. m. pimentel, r. g. teixeira, j. s. valverde (sao paulo ift)
        authors = authors.split(',')
        author_names = [x[:x.find('(')].strip() if '(' in x else x for x in authors]
        aff_names = []
        current_aff = 'unknown'
        for x in reversed(authors):
            if '(' in x:
                current_aff = re.split(' & | and ', format_affiliation_name(x[x.find('(')+1:x.find(')')]))
            aff_names.append(current_aff)
        aff_names = aff_names[::-1]
        # print 'Authors : ', author_names
        # print 'affiliations : ', aff_names
    else:
        author_names = authors.split(',')
        aff_names = ['unknown'] * len(author_names)

    return author_names, aff_names



def format_affiliation_name(affiliation):
    # tokenize:
    affiliation = affiliation.strip().split(' ')
    affiliation = [token for token in affiliation if token not in stpwds]
    affiliation = [token for token in affiliation if token not in countries]
    # normalization: 
    affiliation_formatted=[]
    for token in affiliation:
        if (token in stpwds or token in countries):
            continue
        if token in univ:
            affiliation_formatted.append('univ')
            continue
        if token in dept:
            affiliation_formatted.append('dept')
            continue
        if token in inst:
            affiliation_formatted.append('inst')
            continue
        affiliation_formatted.append(token)   
    
    affiliation = (" ").join(affiliation_formatted)
    return affiliation

def format_author_name(author):
    # remove extra space
    author = author.strip()
    split_author = re.split(r'[ .]+', author)
    if split_author[-1]=='iii' or split_author[-1]=='jr.':
        split_author.pop()

    if len(split_author) <= 1:
        return split_author[0] 
    
    author_name=[] 
    for i in range(len(split_author)-1):
        author_name.append(split_author[i][0]+'.')
    author_name.append(split_author[-1])
    author_name =  ''.join(author_name)
    return author_name


def fix_auth_aff(info):
    list_authors = []
    list_affiliations = []
    flat_affiliations = []
    info['authors'] = info['authors'].replace(np.nan, 'unknown')
    bar = pyprind.ProgBar(len(info['authors']),bar_char='█', width=barwidth)
    for authors in info['authors']:
        if authors != 'unknown':
            # lowercase
            authors = authors.lower().strip(", ")
            # extract the affiliations
            authors, affiliations = parse_authors_affiliations(authors)
            authors_names = []      
            for author in authors:
                if len(author) > 2:
                    authors_names.append(format_author_name(author))
            list_affiliations.append(affiliations)
            flat_affiliations.append(list(flatten_array(affiliations)))
            list_authors.append(authors_names)
        else:
            list_affiliations.append(['unknown'])
            flat_affiliations.append(['unknown'])
            list_authors.append(['unknown'])
        bar.update()   
    info['authors'] = list_authors
    info['affiliations'] = list_affiliations
    info['flat affiliation'] = flat_affiliations

def attribute_features(X, info,train=True):
    
    F = X.copy()
    #flipped = 0
    # number of overlapping words in title
    overlap_title = []
    # temporal distance between the papers
    temp_diff = []
    # number of common authors
    comm_auth = []
    # same journal
    same_journal = []
    # same affiliation (word2vec)
    same_affil = []
    # same affiliation (tfidf)
    same_affil_tfidf = []
    # self citation
    self_citation = []
    if train:
        print 'Computing the tf-ifd of the affiliations'
        TFIDF = affiliations_to_tfidf(info)
        print "Training word2vec for affiliations..."
        model = train_word2vec_affiliations(info)
    else:
        TFIDF = io.loadmat('../data/TFIDF_affiliations')
        TFIDF = TFIDF['TFIDF']
        print TFIDF.shape
        model = word2vec.Word2Vec.load("../data/affil_word2vec_"+str(num_features_affil)+"_4_3") 
    print 'Attribute features...'
    indices = {k: v for(k, v) in [(info.index[i], i) for i in range(0, len(info.index))]}
    bar = pyprind.ProgBar(F.shape[0], bar_char='█', width=barwidth)
    for idx, edge in F.iterrows():
        ids = int(edge['source'])
        idt = int(edge['target'])
        #print ids, "->" ,idt
        source_info = info.loc[ids]
        target_info = info.loc[idt]      
        # convert to lowercase and tokenize
        source_title = source_info["title"].lower().split(" ")
        # remove stopwords
        source_title = [token for token in source_title if token not in stpwds]
        source_title = [stemmer.stem(token) for token in source_title] 
        target_title = target_info["title"].lower().split(" ")
        target_title = [token for token in target_title if token not in stpwds]
        target_title = [stemmer.stem(token) for token in target_title]
        ot = len(set(source_title).intersection(set(target_title)))
        
        source_auth = set(source_info["authors"])
        target_auth = set(target_info["authors"])
        ca = source_auth.intersection(target_auth)
        while 'unknown' in ca: ca.remove('unknown')
        ca = len(ca)

        source_affil = source_info["flat affiliation"]
        target_affil = target_info["flat affiliation"]
        ci = compare_affil(source_affil, target_affil, model, num_features_affil)
        ci_tfidf = np.squeeze(cosine_similarity(TFIDF[indices[ids]],TFIDF[indices[idt]]))

        tdiff = int(source_info["year"]) - int(target_info["year"])
        overlap_title.append(ot)
        temp_diff.append(tdiff)
        comm_auth.append(ca)
        same_affil.append(ci)
        same_affil_tfidf.append(ci_tfidf)
        self_citation.append(ca > 0)
        same_journal.append(source_info["journal"]==target_info["journal"])
        bar.update()
    # Build the set features array
    F['title overlap'] = overlap_title
    F['temporal diff'] = temp_diff
    F['self citation'] = self_citation
    F['same journal'] = same_journal
    F['common authors'] = comm_auth
    F['same affiliation word2vec'] = same_affil
    F['same affiliation tfidf'] = same_affil_tfidf
    
    return F