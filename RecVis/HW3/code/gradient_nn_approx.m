function [grad_s_Wi_approx, grad_s_Wo_approx, grad_s_bi_approx, grad_s_bo_approx] = gradient_nn_approx(X,Y,Wi,bi,Wo,bo,epsilon)

%bo:
[~,~,loss_p]=nnet_forward_logloss(X,Y,Wi,bi,Wo,bo+epsilon);
[~,~,loss_m]=nnet_forward_logloss(X,Y,Wi,bi,Wo,bo-epsilon);
grad_s_bo_approx=(loss_p-loss_m)/2/epsilon;

%Wo:
h=length(Wo);
I=eye(h);
grad_s_Wo_approx=zeros(1,h);
for j=1:h
  [~,~,loss_p]=nnet_forward_logloss(X,Y,Wi,bi,Wo+epsilon*I(j,:),bo);
  [~,~,loss_m]=nnet_forward_logloss(X,Y,Wi,bi,Wo-epsilon*I(j,:),bo); 
  grad_s_Wo_approx(j)=(loss_p-loss_m)/2/epsilon;
end

%bi:
grad_s_bi_approx=zeros(h,1);
for j=1:h
  [~,~,loss_p]=nnet_forward_logloss(X,Y,Wi,bi+epsilon*I(:,j),Wo,bo);
  [~,~,loss_m]=nnet_forward_logloss(X,Y,Wi,bi-epsilon*I(:,j),Wo,bo); 
  grad_s_bi_approx(j)=(loss_p-loss_m)/2/epsilon;
end

%Wi:
p=size(Wi,2);
grad_s_Wi_approx=zeros(h,p);
for j=1:h
    for k=1:p
        E=zeros(h,p);
        E(j,k)=1;
        [~,~,loss_p]=nnet_forward_logloss(X,Y,Wi+epsilon*E,bi,Wo,bo);
        [~,~,loss_m]=nnet_forward_logloss(X,Y,Wi-epsilon*E,bi,Wo,bo); 
        grad_s_Wi_approx(j,k)=(loss_p-loss_m)/2/epsilon;
    end
end

