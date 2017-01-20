function [centers,distortion,affect]=kmeans(X,K,max_iter,tol)
   
  %X: data sample.
  %K: Number of clusters
  %max_iter: maximum number of iterations.
  %tol: Convergence tolerance.
  switch nargin
      case 2
          max_iter=1000;
          tol=1e-4;
      case 3
          tol=1e-4;
  end
      
  %Initialization of the clusers centers:
  [centers,idx]=datasample(X,K,'replace',false);
  %fprintf('Inital centers: %d %d\n',idx);
  distances=zeros(size(X,1),K);
  iter=1;
  delta=+Inf;
  distortion=zeros(1,max_iter);
  while(iter < max_iter && delta>tol)
  %Affectation
  for c =1:K
    distances(:,c)=arrayfun(@(idx) norm((X(idx,:)-centers(c,:))), 1:size(X,1));
    
  end
  [~,affect]=min(distances,[],2);
  %Distortion measure:
  for c=1:K
   distortion(iter)=distortion(iter)+sum(distances(:,c).*(affect==c));
  end
  if iter>1
      delta=distortion(iter-1)-distortion(iter);
  end
  %update centers:
  for c =1:K
    centers(c,:)=mean(X(affect==c,:),1);
  end
  iter=iter+1;
  end
  distortion=distortion(1:iter-1);
  