%% Sum-product algorithm on the given tree:
% X1:{ X2 {X3, X4}, X5}
clearvars; clc
% Conditional probabilities
% The graph nodes and edges:
V = 1:5;
Neighbors = {[2 5], [1 3 4], [2], [2], [1]};
F{1}{2} = [.1 .9;.5 .2;.1 .1]; F{2}{1} = F{1}{2}';
F{2}{3} = [.1 .9 .2 .3;.8 .2 .3 .6]; F{3}{2} = F{2}{3}';
F{2}{4} = [.1 .9;.8 .2];   F{4}{2} = F{2}{4}';
F{1}{5} = [.1 .2;.8 .1;.3 .7]; F{5}{1} = F{1}{5}';
% Computing the repartition function Z
M1 = sum_product(1,Neighbors,F);
M2 = sum_product(2,Neighbors,F);
Z = sum(M1);
assert(Z==sum(M2))
sumM1 = sum_product(1,Neighbors,F)/Z;
sumM2 = sum_product(2,Neighbors,F)/Z;
%% Compute the probability of the graph
P = @(X) F{1}{2}(X(1),X(2))*F{2}{3}(X(2),X(3))*F{2}{4}(X(2),X(4))*F{1}{5}(X(1),X(5))/Z;
P1 = P([1 2 4 2 1])
P2 = P([3 1 2 1 2 ])
P3 = P([2 2 1 2 1])

%% Max Product Algorithm
maxM1 = max_product(1,Neighbors,F)/Z;
maxM2 = max_product(2,Neighbors,F)/Z;