function [MAP_states] = viterbi(K, Pi, U, A, mu,s)
% Inputs:
% Pi: Probabilities for first hidden states
% U: observations
% A: transition matrix
% mu,s: emission probabilities
% K number of possible hidden states

[T,p]=size(U);
%Gaussian emissions:
N=zeros(T,K);
for i=1:K
    for t=1:T
        scalar=(2*pi)^(-p/2)/sqrt(det(s{i}));
        d=U(t,:)-mu(i,:);
        N(t,i)=scalar*exp(-0.5*d*inv(s{i})*d');
    end
end
N=diag(1./sum(N,2))*N;
% Probabilities of paths:
V = zeros(T,K);
P = zeros(T,K);

MAP_states = zeros(T,1);

% Initial probabilities
for i = 1:K
    V(1,i) = Pi(i)*N(1,i);
    P(1,i) = 0;
end

for i = 2:T
    for k = 1:K
    [V(i,k), P(i,k)] = max(V(i-1,:) .* A(:,k)' * N(i,k));
    end
end

% the most likely hidden state:
[~, MAP_states(T)] = max(V(T,:));
for i = T:-1:2
MAP_states(i-1) = P(i,MAP_states(i));
end
end
