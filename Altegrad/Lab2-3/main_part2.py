from __future__ import division
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import operator

#------------------------------------
# Part 2 Robustness of the network
#------------------------------------

# Returns a randomly selected set of nodes
def simple_random_sample(G, num_sample_nodes):
    if num_sample_nodes > G.number_of_nodes():
        print 'Sample size exceeds number of nodes'
        return None
    else:
        sample_nodes = set()
        while len(sample_nodes) < num_sample_nodes:
            sample_nodes.add(random.choice(G.nodes()))
        return sample_nodes


# Simulate random failures: random deletion of nodes
def random_node_deletion (G, fraction_to_delete):
    H = copy.deepcopy(G)
    if fraction_to_delete > 1:
        print 'Trying to delete more nodes than exist in the network'
        return None
    
    number_to_delete = int(round(fraction_to_delete * H.number_of_nodes()))
    nodes_to_delete = simple_random_sample (H, number_to_delete)
    H.remove_nodes_from(nodes_to_delete)
    return H


# Returns a sample of high degree nodes
def degree_biased_sample(G, num_sample_nodes):
    if num_sample_nodes > G.number_of_nodes():
        print 'Sample size exceeds number of nodes'
        return None
    else:
        sorted_deg = sorted(G.degree().items(), key=operator.itemgetter(1))
        sample_nodes = [n[0] for n in sorted_deg[-num_sample_nodes:]]
        return sample_nodes
        

# Simulate targeted failures: targeted deletion of high degree nodes
def targeted_node_deletion(G, fraction_to_delete):
    H = copy.deepcopy(G)
    if fraction_to_delete > 1:
        print 'Trying to delete more nodes than exist in the network'
        return None
    number_to_delete = int(round(fraction_to_delete * H.number_of_nodes()))
    nodes_to_delete = degree_biased_sample(H, number_to_delete)
    H.remove_nodes_from(nodes_to_delete)
    return H


## Main program

# G=nx.read_edgelist("../ca-GrQc.txt", \
# comments='#', delimiter='\t', nodetype=int, create_using=nx.Graph(), \
# encoding='utf-8')

G = nx.fast_gnp_random_graph(1000, 0.01)

G=list(nx.connected_component_subgraphs(G))[0]
G = G.to_undirected()


# Fraction of removed nodes
fraction_del = [float(i)/1000 for i in range(1,301,7)]

# Size of GCC and rest components
r_size_of_GCC_per_iteration = []
r_size_of_other_CC = []
t_size_of_GCC_per_iteration = []
t_size_of_other_CC = []

# Perform random deletion
for i in fraction_del:
    H = random_node_deletion(G, i)
    r_size_of_GCC_per_iteration.append(list(nx.connected_component_subgraphs(H))[0].number_of_nodes() / float(G.number_of_nodes()))
    r_size_of_other_CC.append(1+(H.number_of_nodes() - list(nx.connected_component_subgraphs(H))[0].number_of_nodes()) / float(G.number_of_nodes()))
    del H

# Perform targeted deletion    
for i in fraction_del:
    H = targeted_node_deletion(G, i)
    t_size_of_GCC_per_iteration.append(list(nx.connected_component_subgraphs(H))[0].number_of_nodes() / float(G.number_of_nodes()))
    t_size_of_other_CC.append(1+(H.number_of_nodes() - list(nx.connected_component_subgraphs(H))[0].number_of_nodes()) / float(G.number_of_nodes()))
    del H

# Plot results    
plt.figure('Random-targeted deletions')
plt.plot(fraction_del, r_size_of_GCC_per_iteration, 'bo', label='Random - GCC')
plt.plot(fraction_del, r_size_of_other_CC, 'gs', label='Random - Other CCs')
plt.plot(fraction_del, t_size_of_GCC_per_iteration, 'mo', label='Targeted - GCC')
plt.plot(fraction_del, t_size_of_other_CC, 'ys', label='Targeted - Other CCs')
plt.xlabel('Fraction of deleted nodes')
plt.ylabel('Size of GCC and rest components')

plt.legend(loc=9,ncol=2,bbox_to_anchor=(0., 1, 1., .102),prop={'size':8})
plt.draw()
plt.savefig('robustness_erdos.pdf')
plt.show()    

