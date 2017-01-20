function w = train_cv_linear_svm(features,labels)
% Linear SVM
% Find the best parameters via cross-validation
% Then retrain on the full training set

nsamples = size(features,2);
dim = size(features,1);

Ncosts  = 5;
cost_range  = dim*logsample(dim/100,2*dim,Ncosts);
cv_error=zeros(Ncosts,1);
nfolds = 5;


features = double(features);
labels = double(labels);
if size(labels',2)~=1
    labels = labels';
    features = features';
end
for j=1:Ncosts
    cost   = cost_range(j);
    fprintf('C = %.2e  [%i out of %i]\n',cost,j,Ncosts);
    % use this 'parameter_string' variable as input to 'svmtrain_libsvm'
    parameter_string = sprintf('-s 0 -t 0 -c %.5f',cost); 
    error  =zeros(nfolds,1);
    for k=1:nfolds
        fprintf('.');
        %split data into training set (trset) and validation set (vlset)
        [trset_features,trset_labels,vlset_features,vlset_labels] =  ...
             split_data(features,labels,nsamples,nfolds,k);
        model = svmtrain_libsvm(trset_labels', trset_features', parameter_string);
        % predict on the validation set:
        predict_labels = svmpredict_libsvm(vlset_labels', vlset_features', model);
        error(k)  = sum(predict_labels~=vlset_labels');
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
parameter_string = sprintf('-s 0 -t 0 -c %.5f',cost); 
model = svmtrain_libsvm(labels',features', parameter_string);
w = (model.sv_coef' * full(model.SVs));
w(end) = w(end)-model.rho;