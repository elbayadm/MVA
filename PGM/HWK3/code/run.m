%% Settings
cd '/Users/redgns/GitHub/git-MVA/PGM/HWK3'
addpath(genpath('.'))
clear all
doprint=false;
colors = { 'k' 'r' 'g' 'b' };
markers = { 'o' 's' '+' 'x' };
c=[0 0 0;1 0 0;0 1 0;0 0 1];
set(groot,'defaultAxesColorOrder',c)
iters=50;
% HMM data:
% Observed variable u_t=(x_t,y_t)
% Hidden variable q_t with K=4 possible states

train=load('EMGaussian.data');
test=load('EMGaussian.test');
% Q2
% 
s=load('sigmas.mat');
s=s.sigmas;
mu=load('mus.mat');
mu=mu.mu;
K=4;
Pi=repmat(1/K,1,K);
A=diag(repmat(.5,K,1));
A(A==0)=1/6;
%
[Alpha, Beta, ~,~]=ForwardBackwardMessages(test,K,A,Pi,mu,s);
% Probabilities p(q_t|u1...uT):
p_hidden=Alpha.*Beta;
p_observed=sum(p_hidden,2);
p_hidden=diag(1./p_observed)*p_hidden;
%
figure('paperunits','normalized','paperposition',[0.1 0.1 1.2 .4],'PaperOrientation','landscape'),
subplot(2,1,1)
plot(1:50,p_hidden(1:50,:),'LineWidth',1.5)
legend({'q=1','q=2','q=3','q=4'},'Location','NorthOutside','Orientation','horizontal');
subplot(2,1,2)
plot(51:100,p_hidden(51:100,:),'LineWidth',1.5)
legend({'q=1','q=2','q=3','q=4'},'Location','SouthOutside','Orientation','horizontal');
if doprint
print('-dpdf','images/q100.pdf', '-opengl')
end
%
[mu_HMM,s_HMM,A_HMM,Pi_HMM,ll,ll_test,~]=HMM_EM(K,train,test,A,Pi,s,mu,iters);
%
figure,
plot(1:length(ll),ll,'LineWidth',1.5)
hold on,
plot(1:length(ll),ll_test,'LineWidth',1.5)
hold on,
%Random start points:
rand('seed',117);
for k=1:K
  C=rand(2,2);
  s_rand{k}=(C+C')/2+2*eye(2);
end
mu_rand=rand(K,2);
[mu_HMM_R,s_HMM_R,A_HMM_R,Pi_HMM_R,ll_R,ll_test_R,~]=HMM_EM(K,train,test,A,Pi,s_rand,mu_rand,iters);
plot(1:length(ll_R),ll_R,'LineWidth',1.5)
hold on,
plot(1:length(ll_R),ll_test_R,'LineWidth',1.5)
legend('train-GM','test-GM','train-RI','test-RI','Location','southeast')
xlim([1 length(ll_R)])
h=refline([0,max(ll)]);
set(h,'LineStyle','-.','color','k')
h=refline([0,max(ll_test)]);
set(h,'LineStyle','-.','color','r')
ylabel('Normalized log-likelihood')
xlabel('Iteration')
if doprint
print('-dpdf','images/ll.pdf', '-opengl')
end

%% Comparison to HW2:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EM - isotropic covariances
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

loglikold = -Inf;
mu = mu_rand;            % Random initialization - same as in HMM
sigma2 =   s_rand{1};     % initialize variances with large values
pis = 1/K * ones(K,1);       % initialise with uniform
max_iterations = 1000;
tolerance = 1e-10;
[T,p]=size(train);
for i=1:max_iterations
    % compute distances
    distances = zeros(T,K);
    for k=1:K
        distances(:,k) = sum( ( train - repmat(mu(k,:),T,1) ) .* ( train- repmat(mu(k,:),T,1) ) , 2 );
    end
    % E-step: compute posterior distributions
    logtau_unnormalized = distances;
    for k=1:K
        logtau_unnormalized(:,k) = - .5 * distances(:,k) / sigma2(k) - .5 * p * log( sigma2(k) ) - .5 * p * log(2*pi) + log(pis(k));
    end
    logtau = log_normalize( logtau_unnormalized ); % robust way ( very important in practise )
    tau = exp(logtau);

    % compute log-likelihood
    loglik = ( - sum( logtau(:) .* tau(:) ) + sum( tau(:) .* logtau_unnormalized(:) ) ) / T;
    %fprintf('i = %d - loglik = %e\n',i,loglik);
    % check changes in log likelihood
    if loglik < loglikold - tolerance, error('the distortion is going up!'); end % important for debugging
    if loglik < loglikold + tolerance, break; end
    loglikold = loglik;

    % M-step:
    for k=1:K
        mu(k,:) = sum( repmat( tau(:,k),1 ,p) .* train) / sum( tau(:,k) );
        pis(k) = 1/T * sum(tau(:,k));
        temp = train- repmat( mu(k,:), T , 1 );
        sigma2(k) = sum(  sum( temp.^2, 2) .* tau(:,k) ) / sum( tau(:,k) ) / p;
    end
end
fprintf('Convergence in %d iters\n',i);
loglik_isotropisc = loglik  ;
% compute likelihood on test data
Ttest = size(test,1);
distances = zeros(Ttest,K);
for k=1:K
    distances(:,k) = sum( ( test - repmat(mu(k,:),Ttest,1) ) .* ( test - repmat(mu(k,:),Ttest,1) ) , 2 );
end
% E-step: compute posterior distributions
logtau_unnormalized = distances;
for k=1:K
    logtau_unnormalized(:,k) = - .5 * distances(:,k) / sigma2(k) - .5 * p * log( sigma2(k) )  -.5 * p * log(2*pi) + log(pis(k));
end
logtau = log_normalize( logtau_unnormalized ); % robust way ( very important in practise )
tau = exp(logtau);

% compute log-likelihood
logliktest_isotropisc = ( - sum( logtau(:) .* tau(:) ) + sum( tau(:) .* logtau_unnormalized(:) ) ) / Ttest ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EM - general covariances
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

loglikold = -Inf;
mu = mu_rand;                % random initialization - same as in HMM
sigmas=s_rand;               % same as in HMM
pis = 1/K * ones(K,1);       % initialise with uniform
max_iterations = 1000;
tolerance = 1e-10;

for i=1:max_iterations

    logtau_unnormalized = zeros(T,K);
    for k=1:K
        invSigma = inv( sigmas{k} );
        xc = ( train- repmat(mu(k,:),T,1) );
        logtau_unnormalized(:,k) = - .5 * sum( (xc * invSigma) .* xc , 2 ) - .5 * sum( log( eig( sigmas{k}) ) ) - .5 * p * log(2*pi) + log(pis(k));
    end
    logtau = log_normalize( logtau_unnormalized ); % robust way ( very important in practise )
    tau = exp(logtau);

    % compute log-likelihood
    loglik = ( - sum( logtau(:) .* tau(:) ) + sum( tau(:) .* logtau_unnormalized(:) ) ) / T;
    fprintf('i = %d - loglik = %e\n',i,loglik);
    % check changes in log likelihood
    if loglik < loglikold - tolerance, error('the distortion is going up!'); end % important for debugging
    if loglik < loglikold + tolerance, break; end
    loglikold = loglik;

    % M-step:
    for k=1:K
        mu(k,:) = sum( repmat( tau(:,k),1 ,p) .* train) / sum( tau(:,k) );
        pis(k) = 1/T * sum(tau(:,k));
        temp = train- repmat( mu(k,:), T , 1 );
        sigmas{k} = 1 / sum( tau(:,k) ) * ( temp' * ( repmat( tau(:,k) ,1,p) .* temp ) );
    end
end
fprintf('Convergence in %d iters\n',i);
loglik_general = loglik ;

% compute likelihood on test data
Ttest = size(test,1);
logtau_unnormalized = zeros(Ttest,K);
for k=1:K
    invSigma = inv( sigmas{k} );
    xc = ( test - repmat(mu(k,:),Ttest,1) );
    logtau_unnormalized(:,k) = - .5 * sum( (xc * invSigma) .* xc , 2 ) - .5 * sum( log( eig( sigmas{k}) ) ) - .5 * p * log(2*pi) + log(pis(k));
end
logtau = log_normalize( logtau_unnormalized ); % robust way ( very important in practise )
tau = exp(logtau);

% compute log-likelihood
logliktest_general = ( - sum( logtau(:) .* tau(:) ) + sum( tau(:) .* logtau_unnormalized(:) ) ) / Ttest;

% % normalized
% logliktest_isotropisc
% loglik_isotropisc
% loglik_general
% logliktest_general
%% Recap:
fprintf('Train:\n')
fprintf('Isotropic GM: %.2f\nGeneral GM: %.2f\nHMM from GM: %.2f\nHMM from rand : %.2f\n',loglik_isotropisc,loglik_general,ll(end),ll_R(end));
fprintf('Test:\n')
fprintf('Isotropic GM: %.2f\nGeneral GM: %.2f\nHMM from GM: %.2f\nHMM from rand : %.2f\n',logliktest_isotropisc,logliktest_general,ll_test(end),ll_test_R(end));
%
% plot the train data
[MAP_states] = viterbi(K, Pi_HMM, train, A_HMM, mu_HMM,s_HMM);
R2 = my_chi2inv(.9,2);
figure,
for k=1:K
    ind = find(MAP_states==k);
    plot( train(ind,1),train(ind,2),sprintf('%s%s',colors{k},markers{k}));
    hold on
    draw_ellipse_color( inv(s_HMM{k}) , - s_HMM{k} \ ( mu_HMM(k,:)' ), .5 * mu_HMM(k,:) * ( s_HMM{k} \ ( mu_HMM(k,:)' ) )  - R2/2 ,colors{k})
end
if doprint
print('-dpdf','images/viterbi_train.pdf', '-opengl')
end

%
[Alpha, Beta, N,~]=ForwardBackwardMessages(test,K,A_HMM,Pi_HMM,mu_HMM,s_HMM);
% Probabilities p(q_t|u1...uT):
p_hidden=Alpha.*Beta;
p_observed=sum(p_hidden,2);
p_hidden=diag(1./p_observed)*p_hidden;
%
figure('paperunits','normalized','paperposition',[0.1 0.1 1.2 .4],'PaperOrientation','landscape'),
subplot(2,1,1)
plot(1:50,p_hidden(1:50,:),'LineWidth',1.5)
legend({'q=1','q=2','q=3','q=4'},'Location','NorthOutside','Orientation','horizontal');
subplot(2,1,2)
plot(51:100,p_hidden(51:100,:),'LineWidth',1.5)
legend({'q=1','q=2','q=3','q=4'},'Location','SouthOutside','Orientation','horizontal');
if doprint
print('-dpdf','images/q100_learn.pdf', '-opengl')
end
%
[~,classHMM]=max(p_hidden,[],2);
figure,
for k=1:K
    ind = find(classHMM(1:100)==k);
    plot( test(ind,1),test(ind,2),sprintf('%s%s',colors{k},markers{k}));
    hold on
    draw_ellipse_color( inv(s_HMM{k}) , - s_HMM{k} \ ( mu_HMM(k,:)' ), .5 * mu_HMM(k,:) * ( s_HMM{k} \ ( mu_HMM(k,:)' ) )  - R2/2 ,colors{k})
end
hold off
if doprint
print('-dpdf','images/marginal_class.pdf', '-opengl')
end
%
% plot the test data
[MAP_states] = viterbi(K, Pi_HMM, test(1:100,:), A_HMM, mu_HMM,s_HMM);
figure,
for k=1:K
    ind = find(MAP_states==k);
    plot( test(ind,1),test(ind,2),sprintf('%s%s',colors{k},markers{k}));
    hold on
    draw_ellipse_color( inv(s_HMM{k}) , - s_HMM{k} \ ( mu_HMM(k,:)' ), .5 * mu_HMM(k,:) * ( s_HMM{k} \ ( mu_HMM(k,:)' ) )  - R2/2 ,colors{k})
end
if doprint
print('-dpdf','images/viterbi_test.pdf', '-opengl')
end
fprintf('#Differences : %d\n',sum(MAP_states-classHMM(1:100)))

%% Choice of K:
iters=500;
ll_cv=[];
ll_cv_test=[];
cvg=[];
Ks=2:7;
for K=Ks
rand('seed',117);
Pi=repmat(1/K,1,K);
A=diag(repmat(.5,K,1));
A(A==0)=1/(2*(K-1));
s_rand={};
for k=1:K
  C=rand(2,2);
  s_rand{k}=(C+C')/2+2*eye(2);
end
mu_rand=rand(K,2);
[~,~,~,~,ll_R,ll_test_R,it]=HMM_EM(K,train,test,A,Pi,s_rand,mu_rand,iters);
ll_cv=[ll_cv ll_R(end)];
ll_cv_test=[ll_cv_test ll_test_R(end)];
cvg=[cvg it];
end
figure,
plot(Ks,ll_cv,'-b')
hold on,
plot(Ks,ll_cv_test,'-r')
legend('train','test','Location','SouthEast')
xlabel('K');
ylabel('Log-likelihood');
if doprint
print('-dpdf','images/cv.pdf', '-opengl')
end
cvg