function [v_seq,iter]= centering_step(Q,p,A,b,t,v0,eps)
% Newton method to solve the centering step given:
% the inputs (Q,p,A,b) : Q:nxn, A:mxn, p:nx1 and b:mx1
% the barrier method parameter t 
% starting point  v0 
% tolerance eps

%the Newton step
% log-barrier: phi(v)=-sum(log(b-Av))
% f=t(v'Qv+p'v+phi(v))
% delta=-hessian(f)^(-1).grad(f)
% lambda=grad(f)'.hessian(f)^(-1).grad(f)
max_iter=50;
%Backtracking parameters:
alpha=.1;
beta=.7;
iter=1;
v=v0;
v_seq=v;
while iter<=max_iter
    c=A*v-b;
    grad=t*(2*Q*v+p)-A'*(1./c);
    hessian=2*t*Q;
    for ii=1:size(A,1)
       hessian=hessian+(A(ii,:)'*A(ii,:))./(c(ii)^2);
    end
    delta=-pinv(hessian)*grad;
    lambda=-grad'*delta;
    if(lambda/2) <eps
        %fprintf('Newton converged - %d iterations\n',iter);
       break;
    end
    %Backtracking line search:
    s=1;
    while(barrier_loss(Q,p,A,b,t,v+s*delta) > barrier_loss(Q,p,A,b,t,v)+alpha*s*grad'*delta)
        s=beta*s;
    end
    v=v+s*delta;
    %Store new v:
    v_seq=[v_seq v];
    iter=iter+1;
end


	