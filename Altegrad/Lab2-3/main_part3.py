import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial import distance
from numpy import *
from scipy.cluster.vq import kmeans2
import time
#-----------------------------------------------------------------------------
# 1: Load and plot graph

z=nx.read_edgelist("../karate_graph.edgelist", nodetype=int)

pos=nx.spring_layout(z)
plt.figure(1)
plt.axis('off')
nx.draw_networkx(z,pos)
plt.draw()
plt.savefig('Karate_graph.pdf')

#-----------------------------------------------------------------------------
# 3: Hierarchical clustering:

path_length=nx.all_pairs_shortest_path_length(z) 
n = len(z.nodes())
distances=np.zeros((n,n))

for u,p in path_length.iteritems(): 
	for v,d in p. iteritems ():
		distances[u][v] = d 

hier = hierarchy.average( distances )
plt.figure(2)
hierarchy.dendrogram(hier)
plt.savefig('h_clustering.pdf')

#-----------------------------------------------------------------------------
# 4: Spectral clustering:

def spectralClustering(W, k):
	# Create degree matrix
	D = diag(sum(W, axis=0))
	# Create Laplacian matrix
	L = D - W
	eigval, eigvec = linalg.eig(L) # Calculate eigenvalues and eigenvectors
	eigval = eigval.real # Keep the real part
	eigvec = eigvec.real # Keep the real part
	idx = eigval.argsort() # Get indices of sorted eigenvalues
	eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues
	Y = eigvec[:,:k] # Keep the first k vectors
 	centroids, labels = kmeans2(Y, k)
	return labels

# Assess modularity:
def modularity(G, clustering ):
	clusters = max(clustering) + 1
	modularity = 0 # Initialize total modularity value

	#Iterate over all clusters
	for i in range(clusters):

		# Get the nodes that belong to the i-th cluster (increase node id by 1)
		nodeList = [n for n,v in enumerate(clustering) if v == i]
		
		# Create the subgraphs that correspond to each cluster				
		subG = G.subgraph(nodeList)
		temp1 = nx.number_of_edges(subG) / float(nx.number_of_edges(G))
		temp2 = pow(sum(nx.degree(G, nodeList).values()) / float(2 * nx.number_of_edges(G)), 2)
		modularity = modularity + (temp1 - temp2)
	return modularity

# Apply spectral clustering to the karate dataset
A = nx.adjacency_matrix(z).toarray()
labels = spectralClustering(A, 2)
plt.figure(3)
s='Modularity = '+format(modularity(z,labels),'.2f')
plt.title(s,bbox={'facecolor':'k', 'alpha':0.3, 'pad':10})
nx.draw(z,pos,with_labels=True,node_color=labels,cmap=plt.get_cmap('Paired'))
plt.draw()
plt.savefig('s_clustering.pdf')

# 5: Community evaluation:
# Use the main function as it follows
# Different clustering solutions
clustering = []
clustering.append( [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 , 0 , 0 , 0 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ] )
clustering.append( [ 0 , 1 , 0 , 1 , 0 , 0 , 1 , 0 , 1 , 1 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 1 , 0 , 1 , 0 , 1 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 0 , 1 ] )
clustering.append( [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 ] )

# Compute modularity for each case and print the graph
for i in range(len(clustering )):
	s='Modularity = '+format(modularity(z,clustering[i]),'.2f')
	print s
	plt.figure(4+i)
 	plt.title(s,bbox={'facecolor':'k', 'alpha':0.3, 'pad':10})
	nx.draw(z,pos, with_labels=True,node_color=clustering[i],cmap=plt.get_cmap('Paired'))
	plt.savefig('clustering_'+str(i+1)+'.pdf')

#-----------------------------------------------------------------------------
plt.show()	
#-----------------------------------------------------------------------------


