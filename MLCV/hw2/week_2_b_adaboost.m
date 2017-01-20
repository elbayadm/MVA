% Create the training data
nsamples = 500; 
problem  = 'nonlinear';
[features,labels,posterior] = construct_data(nsamples,'train',problem,'plusminus');
[test_features,test_labels] = construct_data(nsamples,'test',problem,'plusminus');

%------------------------------------------------------------------
% visualize training data and posterior
%------------------------------------------------------------------
figure(2);clf
subplot(1,2,1);
imshow(posterior)
title('P(y=1|X): This is the posterior of  the positive class');
subplot(1,2,2);

pos = find(labels==1);
neg = find(labels~=1);
scatter(features(1,pos),features(2,pos),'r','filled'); hold on,
scatter(features(1,neg),features(2,neg),'b','filled'); 

hold on,axis([0,1,0,1]); axis ij; axis square;
legend({'positives','negatives'},'location','southeast');
title('These are your training data');

%% ------------------------------------------------------------------
% initialize distribution
%------------------------------------------------------------------
[nfeatures,npoints] = size(features);
weights = ones(1,npoints)/npoints;

Rounds_boosting = 400;
f = zeros(Rounds_boosting,npoints);
f_on_grid = 0;
yt = []; ytest =[]; alpha_= []; Z=[];
for it = 1:Rounds_boosting,
    %--------------------------------------------------------
    % Find best weak learner at current round of boosting
    %--------------------------------------------------------
    [coordinate_wl,polarity_wl,theta_wl,err_wl] = best_weak_learner(weights,features,labels);
    yt(it,:) = evaluate_stump(features,coordinate_wl,polarity_wl,theta_wl);
    ytest(it,:) = evaluate_stump(test_features,coordinate_wl,polarity_wl,theta_wl);

    %--------------------------------------------------------
    % estimate alpha
    %--------------------------------------------------------
    alpha_(it) = .5*log((1-err_wl)/err_wl);
    %--------------------------------------------------------
    % update  distribution on inputs 
    %--------------------------------------------------------
     weights = weights.*exp(-alpha_(it)*(yt(it,:).*labels));
     Z(it) = sum(weights);
     weights = weights/Z(it);
    %--------------------------------------------------------
    % compute loss of adaboost at current round
    %--------------------------------------------------------
    final_y = yt'*alpha_';
    % The normalized error upper bound
    err_ub_norm(it) = prod(Z');
    err_ub(it) = sum(exp(-labels'.*final_y));
    % The true error
    err(it) = mean(sign(final_y)~=labels');
    % leave as is - it will produce the classifier images for you
    [weak_learner_on_grid] = evaluate_stump_on_grid([0:.02:1],[0:.02:1],coordinate_wl,polarity_wl,theta_wl);

    %--------------------------------------------------------
    % add current weak learner's response to overall response
    %--------------------------------------------------------
    f_on_grid   = f_on_grid + alpha_(it).*weak_learner_on_grid;
    switch it
        case 10,
            f_10 = f_on_grid;
        case 50,
            f_50 = f_on_grid;
        case 100
            f_100 = f_on_grid;
    end
    
end
predict_test = ytest'*alpha_';
%%
figure(1);clf;
hold on,
plot(err_ub_norm,'linewidth',2)
plot(err,'linewidth',2)
legend({'Error upper bound','True error'},'location','northeast','fontsize',14)
xlabel('Adaboost training round')
ylabel('boosted model normalized error')
box on,
print('-dpdf','figures/loss_adaboost')

figure(4);clf;
plot(err_ub,'linewidth',2)
xlabel('Adaboost training round')
ylabel('boosted model error upper bound')
print('-dpdf','figures/loss_adaboost_bound')

%%
figure(2);clf;
subplot(1,2,1);
imagesc('XData',0:.02:1,'YData',0:.02:1, 'CData',posterior);
box on,
set(gca,'xtick',[],'ytick',[],'Layer', 'top') ; axis image ;

subplot(1,2,2);
imagesc('XData',0:.02:1,'YData',0:.02:1, 'CData',1./(1+exp(-2*f_on_grid)))
box on,
set(gca,'xtick',[],'ytick',[],'Layer', 'top') ; axis image ;
print('-dpdf','figures/classifiers')

figure(3);clf;
colormap gray
subplot(2,2,1);
imagesc('XData',0:.02:1,'YData',0:.02:1, 'CData',f_10,[-1,1]); title('round 10','fontsize',10)
box on,
set(gca,'xtick',[],'ytick',[],'Layer', 'top') ; axis image ;

subplot(2,2,2);
imagesc('XData',0:.02:1,'YData',0:.02:1, 'CData',f_50,[-1,1]); title('round 50','fontsize',10)
box on,
set(gca,'xtick',[],'ytick',[],'Layer', 'top') ; axis image ;

subplot(2,2,3);
imagesc('XData',0:.02:1,'YData',0:.02:1, 'CData',f_100,[-1,1]); title('round 100','fontsize',10)
box on,
set(gca,'xtick',[],'ytick',[],'Layer', 'top') ; axis image ;

subplot(2,2,4);
imagesc('XData',0:.02:1,'YData',0:.02:1, 'CData',f_on_grid,[-1,1]); title('round 400','fontsize',10)
box on,
set(gca,'xtick',[],'ytick',[],'Layer', 'top') ; axis image ;
print('-dpdf','figures/per_round')

%% SVM vs. Adaboost:
%% Performance on the test set
nerrors = sum(sign(predict_test)~=test_labels');
fprintf('Adaboost error rate %.2f%%\n',nerrors/nsamples*100)

%% Decision boundaries:
gamma= 33.5982; cost = 5.4556;
parameter_string = sprintf('-s 0 -t 2 -g %.5f -c %.5f',gamma,cost); 
model = svmtrain_libsvm(labels',features', parameter_string);

[gr_X,gr_Y] = meshgrid(0:.005:1,0:.005:1);
[sv,sh]     = size(gr_X);
coords      = [gr_X(:)';gr_Y(:)'];
dummy       = zeros(1,length(coords));

[~, ~, dec_values]   = svmpredict_libsvm(dummy',coords', model);
values               = reshape(dec_values,[sv,sh]);

tpos = find(test_labels==1);
tneg = find(test_labels~=1);

pos = find(labels==1);
neg = find(labels~=1);

figure(5);clf;
colormap gray
% subplot(1,3,1);
% hold on,
% imagesc('XData',0:.005:1 ,'YData',0:.005:1,'CData',posterior)
% scatter(test_features(1,pos),test_features(2,pos),10,'r','filled');
% scatter(test_features(1,neg),test_features(2,neg),10,'b','filled');
% set(gca,'xtick',[],'ytick',[]) ; axis image ;
% 
% xlim([0 1])
% ylim([0 1])
% 
% title('Ground truth');

subplot(1,2,1);
hold on,
imagesc('XData',0:.005:1 ,'YData',0:.005:1,'CData',values,[-1,1]);
% scatter(test_features(1,tpos),test_features(2,tpos),10,'r','filled');
% scatter(test_features(1,tneg),test_features(2,tneg),10,'b','filled');
scatter(features(1,pos),features(2,pos),10,'r','filled');
scatter(features(1,neg),features(2,neg),10,'b','filled');
set(gca,'xtick',[],'ytick',[]) ; axis image ;
xlim([0 1])
ylim([0 1])
box on,
set(gca, 'Layer', 'top')
title('SVM')

subplot(1,2,2)
hold on,
imagesc('XData',0:.005:1 ,'YData',0:.005:1,'CData',1./(1+exp(-2*f_on_grid)))
%scatter(test_features(1,tpos),test_features(2,tpos),10,'r','filled');
%scatter(test_features(1,tneg),test_features(2,tneg),10,'b','filled');
scatter(features(1,pos),features(2,pos),10,'r','filled');
scatter(features(1,neg),features(2,neg),10,'b','filled');
set(gca,'xtick',[],'ytick',[]) ; axis image ;
xlim([0 1])
ylim([0 1])
title('Adaboost')
box on,
set(gca, 'Layer', 'top')
print('-dpdf','figures/db_train')
