function [mst] = ChowLiu(A,names)

n = size(A,1);
if(nargin < 2)
    names=mat2cell(num2str([1:n]'),ones(n,1));
end
%Remove self loops:
A = A - diag(diag(A));

[v,ind]=max(A(:));
[subi,subj]=ind2sub([n,n],ind);
% Start with node 'subi' and keep track of which nodes are in tree and which are not.
intree = [subi];  number_in_tree = 1;  number_of_edges = 0;
notintree = [1:subi-1,subi+1:n]';  number_notin_tree = n-1;
mst=zeros(n,n);

% Iterate until all n nodes are in tree.
while number_in_tree < n
    %Find the max weight edge from a node that is in tree to one that is not.
    maxcost = 0;                        
    for i=1:number_in_tree,               
    for j=1:number_notin_tree,
      ii = intree(i);  jj = notintree(j);
      if A(ii,jj) > maxcost, 
        maxcost = A(ii,jj); jsave = j; iisave = ii; jjsave = jj;
      end;
    end;
    end;

    % Add this edge and associated node jjsave to 'in tree' and Delete node jsave from 'not in tree'.
    number_of_edges = number_of_edges + 1;      
    mst(iisave,jjsave) = 1;  % Add this edge to tree.
    mst(jjsave,iisave) = 1; 
    number_in_tree = number_in_tree + 1;     
    intree = [intree; jjsave];                 
    for j=jsave+1:number_notin_tree
        notintree(j-1) = notintree(j);
    end
    number_notin_tree = number_notin_tree - 1;
    % %Print update:
    % name1 = mat2str(cell2mat(names(iisave)));
    % name2 = mat2str(cell2mat(names(jjsave)));
    % fprintf('%d %s %d %s\n', iisave, name1, jjsave, name2);
end