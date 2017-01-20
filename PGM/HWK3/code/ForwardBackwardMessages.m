function [Alpha, Beta, N,scales]=ForwardBackwardMessages(U,K,A,Pi,mu,s)
% Outputs:
% Alpha: the forward messages
% Beta: the backward messages
% Inputs:
% U: observed variables
% K: possible hidden states
% A: transition matrix
% Pi: initial probabilities
% mu,s: means and variances of gaussian emissions

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
% Forward:
% We will normalize the alphas and rescale the betas (?12.7 Numerical issues)
scales=zeros(T,1);
Alpha=zeros(T,K);
Alpha(1,:)=Pi.*N(1,:);
scales(1)=sum(Alpha(1,:));
Alpha(1,:)=Alpha(1,:)/scales(1);
for t=2:T
  Alpha(t,:)=(Alpha(t-1,:)*A).*N(t,:);
  scales(t)=sum(Alpha(t,:));
  Alpha(t,:)=Alpha(t,:)/scales(t);
end
% Backward:
Beta=zeros(T,K);
Beta(T,:)=ones(1,K)/scales(T);

for t=T-1:-1:1
    Beta(t,:)=(Beta(t+1,:).*N(t+1,:))*(A')/scales(t); 
end
end
