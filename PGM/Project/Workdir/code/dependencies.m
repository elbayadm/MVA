
%% The context model

disp('Learning th co-occurrences prior...') 
t1=tic;
% Learn the strucure of the tree and parameters for the co-occurences prior
%----------------
% P(b)
%----------------
b=zeros(N,M);
for m=1:M
	if(isfield(train_imdb(m).annotation, 'object'))
        objects = train_imdb(m).annotation.object;
    end
    for o = 1:length(objects)
        name = objects(o).name;
        [flag, index] =ismember(name, names);
        if (flag)
           b(index,m) = 1; 
        end
    end    
end

%Categories stats:
occurences=sum(b,2);
% Evaluate probabilities 
% Joint
prob_bij = zeros(2*N, 2*N);
for i=1:N
    for j=1:i-1
        temp = 2*b(i,:) + b(j,:);
        %Counts of each event:
        p00 = length(find(temp==0)); % P(bi=0,bj=0)
        p01 = length(find(temp==1)); % P(bi=0,bj=1)
        p10 = length(find(temp==2)); % P(bi=1,bj=0)
        p11 = length(find(temp==3)); % P(bi=1,bj=1)
        
        prob_bij([2*(i-1)+1,2*(i-1)+2],[2*(j-1)+1,2*(j-1)+2]) = [p00 p01; p10 p11];
    end
end
prob_bij = prob_bij + prob_bij';
for i=1:N
    p00 = length(find(b(i,:)==0)); % P(bi=0)
    p11 = length(find(b(i,:)==1)); % P(bi=1)   
    prob_bij([2*(i-1)+1,2*(i-1)+2],[2*(i-1)+1,2*(i-1)+2]) = [p00 0; 0 p11];
end
prob_bij = prob_bij / M;

%Evaluate Entropy then Mutual information:
% H(Bi,Bj): sum [-p(bi,bj) log p(bi,bj)]
entropy_bij = zeros(N,N);
tiny= 1e-20;
for i=1:N
    for j=i:N
        pbij = prob_bij(2*i-1:2*i,2*j-1:2*j);
        %Add the tiny value to avoid Infinity
        entropy_bij(i,j) = -pbij(:)'*log(pbij(:)+tiny);
    end
end
% H(Bi): sum [-p(bi) log p(bi)]
entropy_bi = diag(entropy_bij);

% Symmetrize the entropy matrix
entropy_bij=entropy_bij+entropy_bij'-diag(entropy_bi);

% I(Bi,Bj)=H(Bi)+H(Bj)-H(Bi,Bj)
mi = repmat(entropy_bi',N,1)+repmat(entropy_bi,1,N)-entropy_bij;

% Run chow liu's algorithm
t2=tic;
A= ChowLiu(mi,names); 
mst=toc(t2);
chow=toc(t1);
fprintf('Learning dependencies tree in %.3f - of which MST in %.3f\n',chow,mst);
prob_bi = diag(prob_bij);
prob_bi = reshape(prob_bi',2,N)';
% Probability of Bi=1
prob_bi1 = prob_bi(:,2);
A=sparse(A);

%Find message scheduling for inference on a tree

tree_msg = zeros(2*(N-1),2);
tree_msgIndex = N;
prevNodes = [];
currentNodes = root;
while (tree_msgIndex <= 2*(N-1))
  allNextNodes = [];
  for (i = 1:length(currentNodes))
    nextNodes = setdiff(find(A(currentNodes(i),:)),prevNodes);
    Nnext = length(nextNodes);
    tree_msg(tree_msgIndex:tree_msgIndex+Nnext-1,:) = ...
      [repmat(currentNodes(i),Nnext,1), nextNodes'];
    tree_msgIndex = tree_msgIndex + Nnext;
    allNextNodes = [allNextNodes, nextNodes];
  end

  prevNodes = [prevNodes, currentNodes];
  currentNodes = allNextNodes;
end


% Evaluate nodes and edges potentials:
node_potential = ones(N,2);
edge_potential = sparse(2*N,2*N);
node_potential(root,:) = diag(prob_bij(2*root-1:2*root,2*root-1:2*root));

for n=N:2*(N-1)
    i = tree_msg(n,1); % parent
    j = tree_msg(n,2); % child
    p_i = diag(prob_bij(2*i-1:2*i,2*i-1:2*i));
    edge_potential(2*i-1:2*i,2*j-1:2*j) = prob_bij(2*i-1:2*i,2*j-1:2*j)./repmat(p_i,1,2);
end

edge_potential = edge_potential + edge_potential';


% Adapt edge weights for visualization
edge_weight = A.*mi/max(mi(:));
for i=1:N
    for j=i+1:N
        p01 = prob_bij(2*i-1,2*j);
        p10 = prob_bij(2*i,2*j-1);
        p11 = prob_bij(2*i,2*j);
        if(p11 < (p01+p11)*(p10+p11))  % Negative relationship if p(bi=1,bj=1) < p(bi=1)*p(bj=1)
            edge_weight(i,j) = -edge_weight(i,j);
            edge_weight(j,i) = -edge_weight(j,i);
        end
    end
end

disp('Weights stats:')
fprintf('Total edges: %d - Postitive edges: %d - Negative edges: %d\n',full(sum(sum(edge_weight~=0))/2),full(sum(sum(edge_weight>0))/2),full(sum(sum(edge_weight<0))/2));

EE=sort(nonzeros(edge_weight));
[~,I]=sort(abs(EE),'descend');
disp('Top 10 strongest:')
disp(EE(I(1:10))')

if doviz
	% Show the structure of the tree
	subtrees = {'floor','sky','car'}; 
	drawHierarchy(A, names, edge_weight, names, subtrees,1);
end

disp('Nonleaf nodes')
for ii=1:N
    neighbors = find(A(ii,:));
    if(length(neighbors)>1)
        fprintf('%d : %s\n',ii,names{ii});
    end
end
