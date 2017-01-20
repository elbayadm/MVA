function [node_marginals,edge_marginals] = sum_product_binary(A,node_potential,edge_potential,tree_msg)

% Sum-product on a tree with binary variables

% inputs:
%       A = adjacency/transition matrix
%       node_potential(i,:) = node potential at node i
%       edge_potential(2*i-1:2*i,2*j-1:2*j) = edge potential at edge (i,j) (2x2)
%
% output:
%       node_marginals(i,:) = node marginal at node i
%       edge_marginals(2*i-1:2*i,2*j-1:2*j) = edge marginal at edge (i,j)
%
N = size(node_potential,1);
A=logical(A);

msg = ones(N,2*N);  % msg(i,2*j-1:2*j) = message from i to j
in_msg_prod = ones(N,2); % in_msg_prod(i,:) = product of incoming messages
edge_marginals = sparse(2*N,2*N);

for n=1:size(tree_msg,1)
    i = tree_msg(n,1);
    j = tree_msg(n,2);
    
    neighbors = A(i,:);
    neighbors(j) = 0;
    in_msg_prod(i,:) = prod(msg(neighbors,2*i-1:2*i),1);
    msg_ij = node_potential(i,:).*in_msg_prod(i,:);
    msg_ij = repmat(msg_ij',1,2).*edge_potential(2*i-1:2*i,2*j-1:2*j);
    if(sum(msg_ij(:)) > 0)
        msg(i,2*j-1:2*j) = sum(msg_ij,1)/sum(msg_ij(:));
    end
    % Backward pass
    if (n>=size(tree_msg,1)/2) 
        temp = in_msg_prod(i,:)'*in_msg_prod(j,:);  
        temp = temp.*edge_potential(2*i-1:2*i,2*j-1:2*j);
        temp = temp.*(node_potential(i,:)'*node_potential(j,:));
        edge_marginals(2*i-1:2*i,2*j-1:2*j) = temp / sum(temp(:));;
        edge_marginals(2*j-1:2*j,2*i-1:2*i) = temp';
    end
end        

node_marginals = full(sum(edge_marginals,2));
node_marginals = reshape(node_marginals',2,N)';
node_marginals = node_marginals ./ repmat(sum(node_marginals,2),1,2);