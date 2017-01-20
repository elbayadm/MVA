function [w_seq , loss_seq, loss_optim, iter]= subgrad(X,y,lambda,eps,step_policy,h)
% Subgradient (descent) algorithm for LASSO
% X:nxd, y:nx1
% lambda: Regularization parameter
% eps: tolerance

% parameters:
max_iter=10000;
d=size(X,2);
% steps strategy
switch step_policy
    case 'constant'
        step= @(x,k) h;
    case 'length'
        step=@(x,k) h/norm(x);
    case 'sqs'
        step=@(x,k) h/k;
    case 'diminish'
        step=@(x,k) h/sqrt(k);
end
%Initial point (ridge regression optimal):
w=(X'*X+lambda *eye(d))\X'*y;
%w=zeros(d,1);
w_seq=w;
loss_seq=norm(X*w-y)/2+lambda*sum(abs(w));
loss_optim=norm(X*w-y)/2+lambda*sum(abs(w));
for iter=1:max_iter
    g=X'*(X*w-y)+lambda*sign(w);
    w=w-step(g,iter)*g;
    w_seq=[w_seq w];
    loss=norm(X*w-y)/2+lambda*sum(abs(w));
    loss_seq=[loss_seq, loss];
    loss_optim=[loss_optim min(loss,loss_optim(end))];
    if abs(loss_seq(end-1)-loss_seq(end))/abs(loss_seq(end-1)) < eps
        fprintf('Subgradient converged\n');
        break
    end
end