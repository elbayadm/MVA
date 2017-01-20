addpath(genpath('../mlcv_toolbox'))
% Create the training data
nsamples = 500; 
problem  = 'nonlinear';
[features,labels] = construct_data(nsamples,'train',problem,'plusminus');

% display your data
pos = find(labels==1);
neg = find(labels~=1);

figure(1);clf;
scatter(features(1,pos),features(2,pos),'r','filled'); hold on,
scatter(features(1,neg),features(2,neg),'b','filled'); 

%%
Ngammas = 20;
Ncosts  = 20;
gamma_range = logsample(.1,100,Ngammas);
cost_range  = logsample(.1,100,Ncosts);
cv_error=zeros(Ngammas,Ncosts);
nfolds = 10;
for i=1:Ngammas
    gamma = gamma_range(i);
    for j=1:Ncosts
        cost   =cost_range(j);
        fprintf('gamma = %.2f  [%i out of %i],  C = %.2f  [%i out of %i]\n',gamma,i,Ngammas,cost,j,Ncosts);
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
%%
figure(1);clf;
imagesc(cv_error)
ax = gca;
ax.XTick = 2:2:length(gamma_range);
ax.YTick = 2:2:length(cost_range);
ax.XTickLabel = round(gamma_range(ax.YTick),2);
ax.YTickLabel = round(cost_range(ax.YTick),2);
colormap default
print('-dpdf','figures/cv_error');
%% Pick the best gamma and cost - those that minimize the cv_error
%and train an svm using the full training set. 
[~,ind] = min(cv_error(:));
[i,j]   = ind2sub(size(cv_error),ind);
cv_error(i,j)
gamma  = gamma_range(1,i)
cost   = cost_range(1,j)
% train the best model on the full training set
parameter_string = sprintf('-s 0 -t 2 -g %.5f -c %.5f',gamma,cost); 
model = svmtrain_libsvm(labels',features', parameter_string);
predict_labels = svmpredict_libsvm(labels', features', model);
error_train = sum(predict_labels~=labels')
% visualize the model
SVs         = model.SVs;
[gr_X,gr_Y] = meshgrid(0:.005:1,0:.005:1);
[sv,sh]     = size(gr_X);
coords      = [gr_X(:)';gr_Y(:)'];
dummy       = zeros(1,length(coords));

[~, ~, dec_values]   = svmpredict_libsvm(dummy',coords', model);
values               = reshape(dec_values,[sv,sh]);

figure(2);clf;
subplot(1,2,1);
imagesc('XData',0:.005:1,'YData',0:.005:1,'CData',values,[-1,1]);
colormap gray
box on,
set(gca,'xtick',[],'ytick',[],'Layer', 'top') ; axis image ;
xlim([0 1]); ylim([0 1])
subplot(1,2,2);

contour(gr_X,gr_Y,values,[-1.0,0,1.0],'linewidth',2);
hold on,
scatter(SVs(:,1),SVs(:,2),10,'r','filled'); 
box on,
set(gca,'xtick',[],'ytick',[],'Layer', 'top') ; axis image ;
xlim([0 1]); ylim([0 1])
axis equal

print('-dpdf','figures/values_svm');

%% Performance on test set
[test_features,test_labels] = construct_data(nsamples,'test',problem,'plusminus');
pred = svmpredict_libsvm(test_labels',test_features', model);
nerrors = sum(pred~=test_labels');
fprintf('C-SVM error rate %.2f%%\n',nerrors/nsamples*100)