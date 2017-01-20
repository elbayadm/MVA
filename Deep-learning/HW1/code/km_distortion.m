function [distortion,affect]=km_distortion(X,centers)
   
  %X: data sample.
  % centers: k-means centers output
  K=size(centers,1);
  distances=zeros(size(X,1),K);
  for c =1:K
    distances(:,c)=arrayfun(@(idx) norm((X(idx,:)-centers(c,:))), 1:size(X,1));
  end
  [~,affect]=min(distances,[],2);
  
  %Distortion measure:
  distortion=0;
  for c=1:K
   distortion=distortion+sum(distances(:,c).*(affect==c));
  end
  
  