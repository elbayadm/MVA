%% Create the training data
nsamples = 200; 
problem  = 'nonlinear';
[train_features_2D,train_labels] = construct_data(nsamples,'train',problem);

%% display your data
pos = find(train_labels==1);
neg = find(train_labels~=1);

figure(1);clf;
scatter(train_features_2D(1,pos),train_features_2D(2,pos),'r','filled'); hold on,
scatter(train_features_2D(1,neg),train_features_2D(2,neg),'b','filled'); 

%% Apparently the data are not linearly separable. 
%% we therefore use a nonlinear embedding of the features
train_features       = embedding(train_features_2D);
[ndimensions,ndata]  = size(train_features);

%% Some code to visualize the embedding functions
%% evaluate the embedding functions on a regular grid
[grid_x,grid_y] = meshgrid([0:.01:1],[0:.01:1]);
z = embedding([grid_x(:)';grid_y(:)']);
%% show a few of them
figure(1);clf
subplot(2,2,1);
imshow(reshape(z(3,:),[101,101]),[]);
title('\Phi_3(x_1,x_2)'); xlabel('x_1'); ylabel('x_2');
subplot(2,2,2);
imshow(reshape(z(20,:),[101,101]),[]);
title('\Phi_{20}(x_1,x_2)'); xlabel('x_1'); ylabel('x_2');

subplot(2,2,3);
imshow(reshape(z(40,:),[101,101]),[]);
title('\Phi_{40}(x_1,x_2)'); xlabel('x_1'); ylabel('x_2');

subplot(2,2,4);
imshow(reshape(z(121,:),[101,101]),[]);
title('\Phi_{121}(x_1,x_2)'); xlabel('x_1'); ylabel('x_2');

%% Regularized logistic regresssion training of the resulting classifier 
%% using cross-validation
%%
%% generate candidate regularization coefficients, lambda:
%% geometric progression from 0.0001 to 100, in 20 steps
Nlambdas                = 20;
lambda_range            = logsample(0.0001, 100,Nlambdas);
cv_error = [];
% for each of those lambdas
for i=1:Nlambdas
    lambda = lambda_range(1,i);
    
    % perform K-fold cross validation
    K = 10;
    
    errors  =zeros(1,K);
    fprintf('lambda = %.4f  [%i out of %i]\n',lambda,i,Nlambdas);
    
    for validation_run=1:K
        fprintf('.');
        % TEMPLATE FOR CROSS-VALIDATION CODE
        
        %split data into training set (trset) and validation set (vlset)
        [trset_features,trset_labels,vlset_features,vlset_labels] =  ...
            split_data(train_features,train_labels,ndata,K,validation_run);
        
        % train logistic regression @ lambda
        X = trset_features';
        Y = trset_labels';
                        
        [w, score] = logit_newton(X,Y,lambda);
        predicted_label_val        = (1./(1+exp(-w'*vlset_features)) >.5);
        nerrors(1,validation_run)  = sum(predicted_label_val~=vlset_labels);
    end
    fprintf(' \n');
    %The crorss-validation error is the mean of the error
    cv_error(i)=mean(nerrors,2);
end
%%
figure(1);clf;
semilogx(lambda_range,cv_error,'linewidth',2);
xlabel('\lambda')
ylabel('CV error (L_{\lambda})')
print('-dpdf','figures/cv_error');

%% Pick the lambda that minimizes the cross-validation error
[~,i] = min(cv_error);
lambda = lambda_range(i);
%% Retrain using full training set
X = train_features';
Y = train_labels';
[w, score] = logit_newton(X,Y,lambda);
%% visualize the resulting classifier
dense_score = reshape(w'*z,[101,101]);
figure(1);clf;
hold on;
scatter(train_features_2D(1,pos),train_features_2D(2,pos),'r','filled'); 
alpha(.6)
scatter(train_features_2D(1,neg),train_features_2D(2,neg),'b','filled'); 
alpha(.6)
contour(grid_x,grid_y,dense_score,[0,0],'color','k');
axis([0,1,0,1]);
box on,
print('-dpdf','figures/db_logit_reg');

predicted_labels     = (1./(1+exp(-w'*train_features)) >.5);
nerrors= sum(predicted_labels~=train_labels);
fprintf('Regularized logit error (training): %.2f%%\n', nerrors/nsamples*100)

%% evaluate performance on test set
[test_features_2D, test_labels ] = construct_data(nsamples,'test', problem);
test_features        = embedding(test_features_2D);
predicted_labels     = (1./(1+exp(-w'*test_features)) >.5);
nerrors_test= sum(predicted_labels~=test_labels);
fprintf('Regularized logit error (test): %.2f%%\n', nerrors_test/nsamples*100)

post = find(test_labels==1);
negt = find(test_labels~=1);

figure(1);clf; hold on,
scatter(test_features_2D(1,post),test_features_2D(2,post),'r','filled'); hold on,
alpha(.6)
scatter(test_features_2D(1,negt),test_features_2D(2,negt),'b','filled'); 
alpha(.6)
contour(grid_x,grid_y,dense_score,[0,0],'color','k');
axis([0,1,0,1]);
box on,
print('-dpdf','figures/db_logit_reg_test');
