function w = logit_cv_newton(features,labels)
% Logistic regression
% Find the best parameters via cross-validation
% Then retrain on the full training set

nsamples = size(features,2);
dim = size(features,1);
Ncosts  = 10;
cost_range  = logsample(dim/10^6,dim,Ncosts);
cv_error=zeros(Ncosts,1);
nfolds = 5;


features = double(features);
labels = double(labels);

for j=1:Ncosts
    cost   = cost_range(j);
    fprintf('C = %.2e  [%i out of %i]\n',cost,j,Ncosts);
    % use this 'parameter_string' variable as input to 'svmtrain_libsvm'
    error  =zeros(nfolds,1);
    for k=1:nfolds
        fprintf('.');
        %split data into training set (trset) and validation set (vlset)
        [trset_features,trset_labels,vlset_features,vlset_labels] =  ...
             split_data(features,labels,nsamples,nfolds,k);
        [w,~] = logit_newton(trset_features,trset_labels, cost);
        % predict on the validation set:
        sigmoid = @(x) 1./(1+exp(-x));
        vl_scores = sigmoid(w*vlset_features);
        predict_labels = 0+(vl_scores>.5);
        error(k)  = sum(predict_labels~=vlset_labels);
    end
    fprintf(' \n');
    %The generalization error is the mean of the error
    cv_error(j)=mean(error);
end
figure(6),
imagesc(cv_error)
ax = gca;
ax.YTick = 2:2:length(cost_range);
ax.YTickLabel = round(cost_range(ax.YTick),2);
ax.XTick = [];
colormap default
% training on the full training set. 
[~,ind] = min(cv_error);
cost   = cost_range(ind);
[w,~] = logit_newton(trset_features,trset_labels, cost);