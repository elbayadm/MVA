function [ mu_seq, w_seq, loss_seq, dGap, count]= coord_descent(X,y,lambda,eps)

max_iter=1000;
d=size(X,2);
%Define QP parameters:
H=inv(X'*X);
c=H*X'*y;
mu_old=zeros(d,1);
mu_seq=mu_old;
w_seq=H*(X'*y-mu_old);
loss_seq=mu_old'*H*mu_old/2-c'*mu_old;
dGap=lambda*norm(w_seq,1);
count=0;
for iter=1:max_iter
    for coord=1:d
    count=count+1;
    mu=mu_old;
    mu(coord)=(c(coord)-H(coord,:)*mu_old+H(coord,coord)*mu_old(coord))/H(coord,coord);
    mu(coord)=(abs(mu(coord))<=lambda)*mu(coord)+sign(mu(coord))*lambda*(abs(mu(coord))>lambda);
    mu_seq=[mu_seq mu];
    loss_seq=[loss_seq mu'*H*mu/2-c'*mu];
    w_seq=[w_seq H*(X'*y-mu)];
    dGap=[dGap lambda*norm(w_seq(:,end),1)];
    if dGap(end) < eps
        break
    end
    mu_old=mu;
    end
    if abs(loss_seq(end-1)-loss_seq(end))/abs(loss_seq(end-1)) < eps
        fprintf('Coordinate descent converged \n');
        break
    end
    
end
