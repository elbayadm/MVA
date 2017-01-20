from __future__ import division
import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.sparse as sparse
import random
from scipy.stats.stats import pearsonr

doPrint=1
#------------------------------------
# Part 1 Analyzing a Real-World Graph
#------------------------------------
# 1:
G=nx.read_edgelist("../ca-GrQc.txt", \
comments='#', delimiter='\t', nodetype=int, create_using=nx.Graph(), \
encoding='utf-8')
print nx.is_directed(G)
#-----------------------------------------------------------------------------
# 2:
print 'Number of nodes:', G.number_of_nodes() 
print 'Number of edges:', G.number_of_edges() 
print 'Number of connected components:', nx.number_connected_components(G)

# Connected components
GCC=list(nx.connected_component_subgraphs(G))[0]

# Fraction of nodes and edges in GCC 
print "Fraction of nodes in GCC: ", GCC.number_of_nodes() / G.number_of_nodes()
print "Fraction of edges in GCC: ", GCC.number_of_edges() / G.number_of_edges()

#-----------------------------------------------------------------------------
# 3:
# Degree
degree_sequence = G.degree().values()
print "Min degree: ", np.min(degree_sequence)
print "Max degree: ", np.max(degree_sequence)
print "Median degree: ", np.median(degree_sequence)
print "Mean degree: ", np.mean(degree_sequence)

# Degree histogram
y=nx.degree_histogram(G)
plt.figure(1)
plt.plot(y,'b-',marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.draw()
if doPrint:
	plt.savefig("degree.pdf")

plt.figure(2)
plt.loglog(y,'b-',marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.draw()
if doPrint:
	plt.savefig("degree_loglog.pdf")

#-----------------------------------------------------------------------------
# 4:
# (I) Triangles
# Exact number of triangles using the built-in function
t=nx.triangles(GCC)
t_total = sum(t.values())/3
print "Total number of triangles ", t_total

# Approximate triangle counting
# A = nx.adjacency_matrix(GCC).toarray()
# eigval, eigvec = linalg.eig(A)

# Or sparse eigesolver
A = nx.adjacency_matrix(GCC)
num_eigenvalues = A.shape[1]-4000
eigval, eigvec = sparse.linalg.eigs(A.astype('F'), k=num_eigenvalues)
# Plot all eigevalues vs. rank. Obrserve the skeweness and the almost symmetry
# around zero
plt.figure(3)
plt.plot(np.real(eigval), 'b.')
plt.xlabel('Rank')
plt.ylabel('Eigvalues')
plt.draw()
if doPrint:
	plt.savefig('eigval.pdf')

# Spectral counting of triangles
triangles_spectral = np.real(sum(np.power(np.real(eigval), 3)) / 6)

# Plot error vs. rank of approximation
triangles_approx = [(sum(np.power(np.real(eigval[:k]), 3)) / 6) for k in range(0,len(eigval))]
triangles_error = [abs(triangles_approx[k] - triangles_spectral) / float(triangles_spectral) for k in range(0, len(triangles_approx))]

# Plot error vs. rank of approximation
plt.figure(4)
plt.plot(triangles_error, 'b.')
plt.xlabel('Rank of approximation (# of eigenvalues retained)')
plt.ylabel('Error')
plt.draw()
if doPrint:
	plt.savefig("approx_error.pdf")

# Distribution of triangle participation (similar to the degree distribution,
# i.e., in how many triangles each node participates to)
t_values = sorted(set(t.values()))
t_hist = [t.values().count(x) for x in t_values]

plt.figure(5)
plt.loglog(t_values, t_hist, 'bo')
plt.xlabel('Number of triangles')
plt.ylabel('Count')
plt.draw()
if doPrint:
	plt.savefig("triangles.pdf")

# plt.figure('5 bis')
# plt.plot(t_values, t_hist, 'bo')
# plt.xlabel('Number of triangles')
# plt.ylabel('Count')
# plt.draw()
# if doprint:
# 	plt.savefig("triangles_bis.pdf")

# (II) clustering coefficient
avg_clust_coef = nx.average_clustering(GCC)
print "Average clustering coefficient (GCC)", avg_clust_coef

avg_clust_coef = nx.average_clustering(G)
print "Average clustering coefficient (G) ", avg_clust_coef

#-----------------------------------------------------------------------------
# 5:
# Centrality measures

# Degree centrality
deg_centrality = nx.degree_centrality(G)

# Eigenvector centrality
eig_centrality = nx.eigenvector_centrality(G)

# Sort centrality values
sorted_deg_centrality = sorted(deg_centrality.items())
sorted_eig_centrality = sorted(eig_centrality.items())

# Extract centralities
deg_data=[b for a,b in sorted_deg_centrality]
eig_data=[b for a,b in sorted_eig_centrality]

# Compute Pearson correlation coefficient

print "Pearson correlation coefficient ", pearsonr(deg_data, eig_data)

# Plot correlation between degree and eigenvector centrality
plt.figure(6)
plt.plot(deg_data, eig_data, 'ro')
plt.xlabel('Degree centrality')
plt.ylabel('Eigenvector centrality')
plt.draw()
if doPrint:
	plt.savefig("deg_eig_correlation.pdf")

#-----------------------------------------------------------------------------
# 6:
# Generate a random graph with 200 nodes
R = nx.fast_gnp_random_graph(200, 0.1)

print 'Number of nodes:', R.number_of_nodes() 
print 'Number of edges:', R.number_of_edges() 
print 'Number of connected components:', nx.number_connected_components(R)

# Connected components
GCC=list(nx.connected_component_subgraphs(R))[0]

# Fraction of nodes and edges in GCC 
print "Fraction of nodes in GCC: ", GCC.number_of_nodes() / R.number_of_nodes()
print "Fraction of edges in GCC: ", GCC.number_of_edges() / R.number_of_edges()

degree_sequence_random = R.degree().values()
print "Min degree ", np.min(degree_sequence_random)
print "Max degree ", np.max(degree_sequence_random)
print "Median degree ", np.median(degree_sequence_random)
print "Mean degree ", np.mean(degree_sequence_random)

r=nx.degree_histogram(R)

plt.figure(7)
plt.plot(r,'b-',marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.draw()
if doPrint:
	plt.savefig("random_degree.pdf")

# Number of triangles in the random graph
t_R=nx.triangles(R)
t_total_R = sum(t_R.values())/3
print "Total number of triangles ", t_total_R

# Distribution of triangle participation (similar to the degree distribution, i.e., in how many triangles each node participates to)
t_R_values = sorted(set(t_R.values()))
t_R_hist = [t_R.values().count(x) for x in t_R_values]

plt.figure(8)
plt.loglog(t_R_values, t_R_hist, 'bo')
plt.xlabel('Number of triangles')
plt.ylabel('Count')
plt.draw()
if doPrint:
	plt.savefig("triangles_random.pdf")

# Average clustering coefficient - Random graph
avg_clust_coef_R = nx.average_clustering(R)
print "Average clustering coefficient ", avg_clust_coef_R


#-----------------------------------------------------------------------------
# 7:
# Kronecker graphs

# Initiator matrix
A_1 = np.array([[0.99, 0.26], [0.26, 0.53]])
A_G = A_1

# Get Kronecker matrix after k=10 iterations
for i in range(10):
    A_G = np.kron(A_G, A_1)

# Create adjacency matrix
for i in range(A_G.shape[0]):
    for j in range(A_G.shape[1]):
        if random.random() <= A_G[i][j]:
            A_G[i][j] = 1
        else:
            A_G[i][j] = 0
   
# Convert adjacency matrix to graph
G_kron = nx.from_numpy_matrix(A_G, create_using=nx.Graph())

print 'Number of nodes:', G_kron.number_of_nodes() 
print 'Number of edges:', G_kron.number_of_edges() 
print 'Number of connected components:', nx.number_connected_components(G_kron)

# Connected components
GCC=list(nx.connected_component_subgraphs(G_kron))[0]

# Fraction of nodes and edges in GCC 
print "Fraction of nodes in GCC: ", GCC.number_of_nodes() / G_kron.number_of_nodes()
print "Fraction of edges in GCC: ", GCC.number_of_edges() / G_kron.number_of_edges()

degree_sequence_random = G_kron.degree().values()
print "Min degree ", np.min(degree_sequence_random)
print "Max degree ", np.max(degree_sequence_random)
print "Median degree ", np.median(degree_sequence_random)
print "Mean degree ", np.mean(degree_sequence_random)

r=nx.degree_histogram(G_kron)

plt.figure(9)
plt.plot(r,'b-',marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.draw()
if doPrint:
	plt.savefig("Kron_degree.pdf")

# Number of triangles in the  Kronecker graph
t_G_kron=nx.triangles(G_kron)
t_total_G_kron = sum(t_G_kron.values())/3
print "Total number of triangles ", t_total_G_kron

# Distribution of triangle participation (similar to the degree distribution, i.e., in how many triangles each node participates to)
t_G_kron_values = sorted(set(t_G_kron.values()))
t_G_kron_hist = [t_G_kron.values().count(x) for x in t_G_kron_values]

plt.figure(10)
plt.loglog(t_G_kron_values, t_G_kron_hist, 'bo')
plt.xlabel('Number of triangles')
plt.ylabel('Count')
plt.draw()
if doPrint:
	plt.savefig("triangles_kron.pdf")


plt.figure(11)
plt.plot(t_G_kron_values, t_G_kron_hist, 'bo')
plt.xlabel('Number of triangles')
plt.ylabel('Count')
plt.draw()
if doPrint:
	plt.savefig("triangles_kron_bis.pdf")

# Average clustering coefficient - Kronecker graph
avg_clust_coef_G_kron = nx.average_clustering(G_kron)
print "Average clustering coefficient ", avg_clust_coef_G_kron

#---------------------------------------------------------
plt.show()