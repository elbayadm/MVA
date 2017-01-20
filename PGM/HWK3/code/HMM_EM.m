function [mu,s,A,Pi,ll,ll_test,it]=HMM_EM(K,U,U_test,A,Pi,s,mu,iters)
% Extra outputs:
% ll: log-likelihood evolution of the train set.
% ll_test: log-likelihood evolution of the test set.
% it: Required iterations for convergence
% Inputs/outputs:
% U: observed variables (train)
% U_test : observed test set
% K: possible hidden states
% A: transition matrix
% Pi: initial probabilities
% m,s: means and variances of gaussian emissions
% iters: max iterations
tol=1e-10;
T=size(U,1);
ll=[];
ll_test=[];
for it=1:iters
    
   %(E)step:
   [Alpha, Beta, N,scales]=ForwardBackwardMessages(U,K,A,Pi,mu,s);
   % Probabilities p(q_t|u1...uT):
    p_hidden=Alpha.*Beta;
    % Probabilities p(u1...uT):
    p_observed=sum(p_hidden,2);
    p_hidden=diag(1./p_observed)*p_hidden;
    % Probabilities p(q_t,q_t+1|u1...uT):
    p_hidden_joint=zeros(K,K,T-1);
    for i=1:T-1
        p_hidden_joint(:,:,i)=A.*( Alpha(i,:)' * (Beta(i+1,:).*N(i+1,:)))/p_observed(i);
    end
    %(M)step:
    Pi=p_hidden(1,:);
    A=sum(p_hidden_joint,3);
    A=diag(1./sum(A,2))*A;
    mu=diag(1./sum(p_hidden,1))*(p_hidden'*U);
    for k=1:K
        d=(U-ones(T,1)*mu(k,:));
        s{k}=d'*(repmat(p_hidden(:,k),1,2).*d)/sum(p_hidden(:,k));
    end
    ll=[ll sum(log(scales))];
    fprintf('Iter %d  Log-likelihood: %.3f\n',it,ll(end));
    % On the test set:
    ll_test=[ll_test hmmlogl(U_test,K,mu,s,A,Pi)];
    if length(ll)>=2 && (ll(end)-ll(end-1))<tol 
        fprintf('Optimization done\n');
        break;
    end
end
ll=ll/T;
ll_test=ll_test/size(U_test,1);

end