function [v_seq, loss, dGap, newton]=barr_method(Q,p,A,b,v0,eps,mu)
% Barrier method to solve QP:
% the inputs (Q,p,A,b) : Q:nxn, A:mxn, p:nx1 and b:mx1
% the barrier method parameter t 
% starting point  v0 
% tolerance eps

%Parameters:
t=1;
max_iter=1000;
m=size(A,1);
v=v0;
v_seq=v;
loss=v'*Q*v+p'*v;
dGap=[];
newton=[];
% Centering step:
for iter=1:max_iter
    %fprintf('iteration %d\n',iter);
    [new_v,nt]=centering_step(Q,p,A,b,t,v,eps);
    newton=[newton, nt];
    v=new_v(:,end);
    v_seq=[v_seq v];
    loss=[loss v'*Q*v+p'*v];
    dGap=[dGap,m/t];
    if (m/t)<eps
      %fprintf('Optimum found\n');
      break;
    else
    t=mu*t;
    
    end
end
