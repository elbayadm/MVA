function [w_seq,loss_seq, iter]=prox_acc_method(X,y,lambda,eps)

d=size(X,2);
%spectral radius
M=max(abs(eig(X'*X)));
max_iter=10;
w_seq=zeros(d,1);
loss_seq=norm(X*w_seq-y)^2/2+lambda*sum(abs(w_seq));
step=@(k) (k>0)*(k-1)/(k+2);
v=w_seq;
w=w_seq;
for iter=1:max_iter
    c=v-X'*(X*v-y)/M
    for ii=1:d
       w(ii)=(M*abs(c(ii))>1)*(c(ii)-sign(c(ii))/M);
    end
    w
    v=w+step(iter)*(w-w_seq(:,end))
    w_seq=[w_seq w];
    loss_seq=[loss_seq norm(X*w-y)^2/2+lambda*sum(abs(w))]; 
   if abs(loss_seq(end-1)-loss_seq(end))/abs(loss_seq(end-1)) < eps
       fprintf('Proximal accelerated method converged\n');
       break
   end
end


