function varargout = show_misclassified(pred, true, X, n, sz)
% Show some misclassified samples 
Mis = find(true ~=pred);
n = min(length(Mis),n);
for i=1:n
vl_tightsubplot(n,i,'box','inner') ;
imshow(reshape(X(Mis(i),1:(sz(1)*sz(2))),sz)) ;
text(2,25,sprintf('pred = %d',pred(Mis(i))),...
       'background','w','fontsize',12) ;
set(gca,'xtick',[],'ytick',[]) ; axis image ;
end
varargout{1} = Mis;
