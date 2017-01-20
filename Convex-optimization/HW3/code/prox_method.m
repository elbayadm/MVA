function [w_seq,loss_seq, iter]=prox_method(X,y,lambda,eps)

d=size(X,2);
%spectral radius
M=max(abs(eig(X'*X)));
max_iter=1000;
w=zeros(d,1);
w_seq=w;
loss_seq=norm(X*w-y)^2/2+lambda*sum(abs(w));
for iter=1:max_iter
   w=w-(X'*(X*w-y))/M;
   for ii=1:d
       w(ii)=(M*abs(w(ii))>1)*(w(ii)-sign(w(ii))/M);
   end
   w_seq=[w_seq w];
   loss_seq=[loss_seq norm(X*w-y)^2/2+lambda*sum(abs(w))];
   if abs(loss_seq(end-1)-loss_seq(end))/abs(loss_seq(end-1)) < eps
       fprintf('Proximal method converged\n');
       break
   end
end


