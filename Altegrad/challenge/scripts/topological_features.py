#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''

Build the graphs and compute the topological features:
1. Dispersion
2. Nb. common neighbors
3. Jaccard coefficient
4. Adamic/Adar coefficient
5. Difference of inlinks
6. Nb. of articles that cited the target
7. Max cited author pagerank (authors citations graph)
8. Are articles in the same community
9. Are authors in the same community

'''
import csv
import numpy as np
import networkx as nx
from networkx.algorithms import approximation as approx
from math import log
import itertools
from collections import defaultdict
from scipy.cluster.vq import kmeans2
import community as co
import pyprind
from time import time
#from parallel_betweenness import betweenness_centrality_parallel
from utils import *

def build_graph(Xtrain, Ytrain, info):
	"""
	Citations graph (main graph)
	"""
	bar = pyprind.ProgBar(len(Xtrain),bar_char='█', width=barwidth)
	pairs = [] 
	for i in range(len(Xtrain)):
		if Ytrain.values[i] ==1:
			pairs.append([Xtrain.values[i,0],Xtrain.values[i,1]])
		bar.update()
	print 'Adding citations edges to the graph'
	G = nx.DiGraph(pairs)
	# Add the remaining nodes:
	G.add_nodes_from(info.index)
	print 'Number of nodes:', G.number_of_nodes() 
	print 'Number of edges:', G.number_of_edges() 
	return G  

def build_authors_citation_graph(Xtrain, Ytrain, info):
	"""
	Authors citations graph
	"""
	print 'Building the authors citations graph'
	GAC = nx.DiGraph()
	unique_authors = flatten_array(info['authors'])
	if 'unknown' in unique_authors:
		unique_authors.remove('unknown')
	GAC.add_nodes_from(unique_authors)
	auth_citations = defaultdict(int)
	bar = pyprind.ProgBar(len(Xtrain), bar_char='█', width=barwidth)
	for i in range(len(Xtrain)):
		if Ytrain.values[i]==1:
			ids = int(Xtrain.values[i,0])
			idt = int(Xtrain.values[i,1])
			source_info = info.loc[ids]
			target_info = info.loc[idt]
			source_auth = set(source_info["authors"])
			target_auth = set(target_info["authors"])
			for sa, ta in itertools.product(source_auth, target_auth):
				if (sa != 'unknown' and ta != 'unknown'):
					auth_citations[sa, ta] += 1
		bar.update()
	for k,v in auth_citations.iteritems():
		GAC.add_edge(k[0], k[1], weight = v)
	# for line in nx.generate_edgelist(GAC):
	# 	print(line)
	print '[GAC] Number of nodes:', GAC.number_of_nodes() 
	print '[GAC] Number of edges:', GAC.number_of_edges() 
	return GAC


def build_authors_coauthorship_graph(info):
	"""
	Authors co-authorship graph
	"""
	print 'Building the co-authorship graph'
	GAA = nx.DiGraph()
	unique_authors = flatten_array(info['authors'])
	if 'unknown' in unique_authors:
		unique_authors.remove('unknown')
	GAA.add_nodes_from(unique_authors)
	auth_coauthorship = defaultdict(int)
	bar = pyprind.ProgBar(len(info['authors']), bar_char='█', width=barwidth)
	for authors in info['authors']:
		bar.update()
		for u, v in itertools.product(authors, authors):
			if (u!=v and u!='unknown' and v!='unknown'):
				auth_coauthorship[u, v] += 1
	for k,v in auth_coauthorship.iteritems():
		GAA.add_edge(k[0], k[1], weight = v)
	print '[GAA] Number of nodes:', GAA.number_of_nodes() 
	print '[GAA] Number of edges:', GAA.number_of_edges() 
	return GAA


def get_authors_rank(Edges,GAC,info,train=True):
	""" 
	compute authors rank and max 'to' authority on authors-citation graph (GAC)
	"""
	print 'Computing authors ranks'
	target_authority = []
	if train:
		t0 = time()
		scores = nx.pagerank(GAC,tol=1e-03)
		p_time = time() - t0
		print("Pagerank runtime: %0.3fs" % p_time)
		np.save('../data/GAC_ranks',scores)
	else:
		scores = np.load('../data/GAC_ranks.npy')
	print 'Retrieving cited authors ranks...'
	bar = pyprind.ProgBar(len(Edges), bar_char='█', width=barwidth)
	for e in Edges[:]:
		idt = int(e[1])
		target_info = info.loc[idt]
		target_auth = set(target_info["authors"])
		target_ranks = [scores[k] if k !='unknown' else 0 for k in target_auth]
		target_authority.append(max(target_ranks))
		bar.update()
	return target_authority


def get_authors_similariy(Edges, GAA, info, train=True):
	""" 
	compute authors similarity on authors-co-authorship graph (GAA) 
	"""
	print 'Computing authors similarity...'
	
	if train:
		t0 = time()
		sim ,idx = simrank(GAA, c=0.8, max_iter=100, eps=1e-4)
		s_time = time() - t0
		print("Simrank runtime: %0.3fs" % s_time)
		np.save('../data/GAA_similarity',sim)
	else:
		sim = np.load('../data/GAA_similarity.npy')
		nodes = GAA.nodes()
		idx = {k: v for(k, v) in [(nodes[i], i) for i in range(0, len(nodes))]}

	print 'Retrieving citing and cited authors similarities...'
	similarity = []
	bar = pyprind.ProgBar(len(Edges), bar_char='█', width=barwidth)
	for e in Edges[:]:
		ids = int(e[0])
		idt = int(e[1])
		source_info = info.loc[ids]
		source_auth = set(source_info["authors"])
		while 'unknown' in source_auth: source_auth.remove('unknown')    

		target_info = info.loc[idt]
		target_auth = set(target_info["authors"])
		while 'unknown' in target_auth: target_auth.remove('unknown')
		if len(source_auth) and len(target_auth):
			st_sim = [sim[idx[u],idx[v]] for u,v in itertools.product(source_auth, target_auth)]
			st_sim = max(st_sim)
		else:
			st_sim = 0
		similarity.append(st_sim)
		bar.update()
	return similarity

def get_dispersion(Edges,G):
	print 'Computing edge nodes dispersion'
	return [nx.dispersion(G,u=e[0],v=e[1]) for e in Edges[:]]

def get_betweenness_centrality(Edges, G, train=True):    
	"""
	Difference in betweenness centrality
	"""
	print 'Computing betweenness centrality'
	#c = nx.degree_centrality(G)
	if train:
		n = len(G.nodes())
		t0 = time()
		#c = betweenness_centrality_parallel(G,processes=4)
		c = nx.betweenness_centrality(G,k=100)
		b_time = time()-t0
		print("Betweenness centrality runtime: %0.3fs" % b_time)
		w = csv.writer(open('../data/G_betweenness.csv', "w"))
		for key, val in c.items():
			w.writerow([key, val])
	else:
		c = {}
		for key, val in csv.reader(open('../data/G_betweenness.csv')):
			c[int(key)] = val

	bc = [c[e[1]] - c[e[0]] for e in Edges[:]]
	return np.array(bc)

def get_common_neighbors(Edges, G): 
	"""
	Number of shared neighbors
	"""
	print 'Computing common neighbors'
	GU = G.to_undirected()
	#common_neighbors = [len(set(GU.neighbors(e[0])).intersection(set(GU.neighbors(e[1])))) for e in Edges[:]]
	common_neighbors = [nx.common_neighbors(GU, e[0], e[1]) for e in Edges[:]]
	cn = [len(list(n)) for n in common_neighbors]
	return cn

def get_jaccard_coeffs(Edges, G): 
	"""
	Jaccard coefficients
	"""
	print 'Computing Jaccard coefficients'
	GU = G.to_undirected()
	JC = nx.jaccard_coefficient(GU, [(e[0],e[1]) for e in Edges[:]])
	jaccard = [jac for u, v, jac in JC]
	return np.array(jaccard)

def get_diff_inlinks(Edges, G):
	print 'Computing difference in in-links'
	in_degrees = G.in_degree()
	diff_deg = [in_degrees[e[1]] - in_degrees[e[0]] for e in Edges[:]]
	to_cited = [in_degrees[e[1]] for e in Edges[:]]
	return diff_deg, to_cited


def get_same_article_community(Edges,G,train=True):
	print 'Verifying if articles are in the same community'
	if train:
		t0 = time()
		GU = G.to_undirected()
		partition = co.best_partition(GU)
		print 'Partition modularity : ', co.modularity(partition,GU)
		co_time = time() - t0
		print("Community detection runtime: %0.3fs" % co_time)
		w = csv.writer(open('../data/G_partition.csv', "w"))
		for key, val in partition.items():
			w.writerow([key, val])
	else:
		partition = {}
		for key, val in csv.reader(open('../data/G_partition.csv')):
			partition[int(key)] = val
	same_co = [partition[e[0]]== partition[e[1]] for e in Edges[:]]
	return same_co

def get_same_authors_community(Edges,GAC, info,train=True):
	print 'Verifying if authors are in the same community'
	if train:
		t0 = time()
		GACU = GAC.to_undirected()
		partition = co.best_partition(GACU)
		print 'Partition modularity : ', co.modularity(partition,GACU)
		co_time = time() - t0
		print("Community detection runtime: %0.3fs" % co_time)
		w = csv.writer(open('../data/GAC_partition.csv', "w"))
		for key, val in partition.items():
			w.writerow([key, val])
	else:
		partition = {}
		for key, val in csv.reader(open('../data/GAC_partition.csv')):
			partition[key] = val
	bar = pyprind.ProgBar(len(Edges),bar_char='█', width=barwidth)
	same_co = []
	for e in Edges[:]:
		bar.update()
		ids = int(e[0])
		idt = int(e[1])
		source_info = info.loc[ids]
		source_auth = set(source_info["authors"])
		while 'unknown' in source_auth: source_auth.remove('unknown')    

		target_info = info.loc[idt]
		target_auth = set(target_info["authors"])
		while 'unknown' in target_auth: target_auth.remove('unknown')
		if len(source_auth) and len(target_auth):
			s = [partition[u]==partition[v] for u,v in itertools.product(source_auth, target_auth)]
			s = any(s)
		else:
			s = False
		same_co.append(s)
	return same_co

def get_adamic_adar(G, Edges):
	"""
	Adamic/adar coefficient
	"""
	GU = G.to_undirected()
	common_neighbors = [nx.common_neighbors(GU, e[0], e[1]) for e in Edges[:]]
	adar = []
	print 'Computing adamic/Adar coefficients'
	bar = pyprind.ProgBar(len(common_neighbors),bar_char='█', width=barwidth)
	for cn in common_neighbors:
		bar.update()
		NCN = [GU.degree(node) for node in cn]
		adar.append(sum(1/log(n+1) for n in NCN))
	return adar	

def get_adamic_adar_authors(GAA, Edges, info):
	"""
	Adamic/adar coefficient
	"""
	print 'Computing adamic/Adar coefficients in the authors graph'
	bar = pyprind.ProgBar(len(Edges),bar_char='█', width=barwidth)
	adar_authors = []
	GU = GAA.to_undirected()
	for e in Edges[:]:
		bar.update()
		ids = int(e[0])
		idt = int(e[1])
		source_info = info.loc[ids]
		source_auth = set(source_info["authors"])
		while 'unknown' in source_auth: source_auth.remove('unknown')    

		target_info = info.loc[idt]
		target_auth = set(target_info["authors"])
		while 'unknown' in target_auth: target_auth.remove('unknown')
		if len(source_auth) and len(target_auth):
			adar = []
			for u,v in itertools.product(source_auth, target_auth):
				common_neighbors = nx.common_neighbors(GU, u, v)
				NCN = [GU.degree(node) for node in common_neighbors]
				adar.append(sum(1/log(n+1) for n in NCN))
			adar_authors.append(max(adar))
		else:
			adar_authors.append(0.0)
	return adar_authors

def simrank(G, c=0.8, max_iter=100, eps=1e-4):
	"""
	Simrank algorithm
	"""
	nodes = G.nodes()
	idx = {k: v for(k, v) in [(nodes[i], i) for i in range(0, len(nodes))]}
	sim_prev = np.zeros(len(nodes))
	sim = np.identity(len(nodes))
	for i in range(max_iter):
		print "Simrank iteration ", i+1
		if np.allclose(sim, sim_prev, atol=eps, rtol=0):
			break
		sim_prev = sim.copy()
		for u, v in itertools.product(nodes, nodes):
			if u is v:
				continue
        	u_ps, v_ps = G.neighbors(u), G.neighbors(v)
        	if len(u_ps)== 0 or len(v_ps) == 0:
        		sim[idx[u]][idx[v]] = 0
        	else:
        		# evaluating the similarity of current iteration nodes pair
        		s_uv = sum([sim_prev[idx[u_p]][idx[v_p]] for u_p, v_p in itertools.product(u_ps, v_ps)])
        		sim[idx[u]][idx[v]] = (c * s_uv) / (len(u_ps) * len(v_ps))
	return sim, idx

def spectral_clustering(G, k):
	GU = G.to_undirected()
	A = nx.adjacency_matrix(GU).toarray()
	# Create degree matrix
	D = np.diag(np.sum(A, axis=0))
	# Create Laplacian matrix
	L = D - A
	eigval, eigvec = np.linalg.eig(L) # Calculate eigenvalues and eigenvectors
	eigval = eigval.real # Keep the real part
	eigvec = eigvec.real # Keep the real part
	idx = eigval.argsort() # Get indices of sorted eigenvalues
	eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues
	Y = eigvec[:,:k] # Keep the first k vectors
 	centroids, labels = kmeans2(Y, k)
	return labels

def topologic_features(X, G, GAC, GAA, info, train=True):
	print 'Topologic features...'
	F = X.copy()
	Edges = X.values
	F['dispersion'] 		   = get_dispersion(Edges, G)
	F['common neighbours']     = get_common_neighbors(Edges, G)
	F['Jaccard coefficient']   = get_jaccard_coeffs(Edges, G)
	F['Adar']                  = get_adamic_adar(G, Edges)
	F['Adar authors']        = get_adamic_adar_authors(GAA, Edges, info)
	#F['Adar authors 2']        = get_adamic_adar_authors(GAC, Edges, info)
	diff_deg, to_cited         = get_diff_inlinks(Edges, G)
	F['diff inlinks']          = diff_deg
	F["to cited"]              = to_cited 
	F['cited authority']       = get_authors_rank(Edges,GAC,info)
	F['articles same co']      = get_same_article_community(Edges,G,train)
	F['authors same co']       = get_same_authors_community(Edges,GAC, info,train)
 	return F
