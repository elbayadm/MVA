function w = train_cv_rbf_svm(features,labels)
% RBF SVM
% Find the best parameters via cross-validation
% Then retrain on the full training set

nsamples = size(features,2);
dim = size(features,1);

Ngammas = 5;
Ncosts  = 5;
gamma_range = logsample(3/dim,100/dim,Ngammas);
cost_range  = logsample(dim/100,2*dim,Ncosts);
cv_error=zeros(Ngammas,Ncosts);
nfolds = 5;
features = double(features);
labels = double(labels);
for i=1:Ngammas
    gamma = gamma_range(i);
    for j=1:Ncosts
        cost   = cost_range(j);
        fprintf('gamma = %.2e  [%i out of %i],  C = %.2e  [%i out of %i]\n',gamma,i,Ngammas,cost,j,Ncosts);
        % use this 'parameter_string' variable as input to 'svmtrain_libsvm'
        parameter_string = sprintf('-s 0 -t 2 -g %.5f -c %.5f',gamma,cost); 
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
        cv_error(i,j)=mean(error);
    end
end
figure(6),
imagesc(cv_error)
ax = gca;
ax.XTick = 2:2:length(gamma_range);
ax.YTick = 2:2:length(cost_range);
ax.XTickLabel = round(gamma_range(ax.YTick),2);
ax.YTickLabel = round(cost_range(ax.YTick),2);
colormap default
% training on the full training set. 
[~,ind] = min(cv_error(:));
[i,j]   = ind2sub(size(cv_error),ind);
gamma  = gamma_range(1,i);
cost   = cost_range(1,j);
parameter_string = sprintf('-s 0 -t 2 -g %.5f -c %.5f',gamma,cost); 
model = svmtrain_libsvm(labels',features', parameter_string);
w = (model.sv_coef' * full(model.SVs));
w(end) = w(end)-model.rho;