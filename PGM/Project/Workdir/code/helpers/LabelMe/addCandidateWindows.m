function [adjmat, node_pot, edge_pot, pw1lscore] = addCandidateWindows(obj,scores,windowScore, A, node_potential_g, edge_potential, prob_bi1)

% Add candidate windows as measurements to the context model
%
% PARAMETERS: 
%       obj(k) = object index for window k
%       scores(k) = detector score for window k
%       windowScore = measurement model trained from the training set
%       A = prior adjacency matrix
%       node_potential_g = prior node potential
%       edge_potential = prior edge potential
%       prob_bi1(i) = P(object i present)
%
% OUTPUTS:
%       adjmat = adjacency matrix of the joint (prior + measurement) model
%       node_pot = node potential
%       edge_pot = edge potential
%       pw1lscore(k) = P(window k correct detection | scores(k))
%
% 2010 April, Myung Jin Choi, MIT.


meas_adjmat = sparse(N, W);
meas_edge_pot = sparse(2*N,2*W);
meas_node_pot = zeros(W,2);
pw1lscore = zeros(W,1);
numWindows = zeros(N,1);

for ww=1:W   
    i = obj(ww);
    numWindows(i) = numWindows(i)+1;
    meas_adjmat(i,ww) = 1;
    pw1lb1 = windowScore.pKthCorrectGivenObjectPresent{i}(numWindows(i)); % P(ww th window correct | bi=1)
    meas_edge_pot(2*i-1:2*i,2*ww-1:2*ww) = [1 0; 1-pw1lb1, pw1lb1];
    pw1lscore(ww) = glmval(windowScore.logitCoef{i}, scores(ww), 'logit');
    pCorrect = pw1lb1*prob_bi1(i);
    meas_node_pot(ww,:) = [(1-pw1lscore(ww))/(1-pCorrect) pw1lscore(ww)/pCorrect];
end

for i=1:N %For consistency
    node_potential_g(i,2) = node_potential_g(i,2)*prod(1-windowScore.pKthCorrectGivenObjectPresent{i}(numWindows(i)+1:end));
end
    
adjmat = [A meas_adjmat; meas_adjmat' sparse(W,W)];
node_pot = [node_potential_g; meas_node_pot];
edge_pot = [edge_potential meas_edge_pot; meas_edge_pot' sparse(2*W,2*W)];
node_pot = node_pot./repmat(sum(node_pot,2),1,2);