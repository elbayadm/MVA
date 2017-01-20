function ll=hmmlogl(U,K,mu,s,A,Pi)
% Similar to ForwardBackwardMessages with only the forward step to compute log-likelihood.    
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
% We will normalize the message to avoid converging to zeros
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
ll=sum(log(scales));
end
