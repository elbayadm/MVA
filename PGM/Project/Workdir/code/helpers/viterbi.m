function map = viterbi(A,node_potential,edge_potential,tree_msg)

% iterbi ~ max-product on a tree with binary variables

% inputs:
%       A = adjacency/transition matrix
%       node_potential(i,:) = node potential at node i
%       edge_potential(2*i-1:2*i,2*j-1:2*j) = edge potential at edge (i,j) (2x2)
%
% output:
%       map(i) = MAP estimate at node i
%

N = size(node_potential,1);
A=logical(A);

msg = sparse(N,2*N);  % msg(i,2*j-1:2*j) = message from i to j
states = sparse(N,2*N); % states(i,2*j-1:2*j) = maximizing state of node i for each value of node j
in_msg_prod = zeros(N,2);  % in_msg_prod(i,:) = product of incoming tree_msg

% Forward pass
for n=1:size(tree_msg,1)/2
    i = tree_msg(n,1);
    j = tree_msg(n,2);
    neighbors = A(i,:);
    neighbors(j) = 0;
    in_msg_prod(i,:) = prod(msg(neighbors,2*i-1:2*i),1);
    msg_ij = node_potential(i,:).*in_msg_prod(i,:);
    msg_ij = repmat(msg_ij',1,2).*edge_potential(2*i-1:2*i,2*j-1:2*j);
    [msg(i,2*j-1:2*j), states(i,2*j-1:2*j)] = max(msg_ij,[],1);
end

map = zeros(N,1);
% Max state of root
neighbors = A(j,:);
in_msg_prod(j,:) = prod(msg(neighbors,2*j-1:2*j),1);
root_marginals = node_potential(j,:).*in_msg_prod(j,:);
[~, map(j)] = max(root_marginals);

% Backward pass
for n=N:2(N-1)
    i = tree_msg(n,1);
    j = tree_msg(n,2);
    map(j) = states(j,2*(i-1)+map(i));
end        
