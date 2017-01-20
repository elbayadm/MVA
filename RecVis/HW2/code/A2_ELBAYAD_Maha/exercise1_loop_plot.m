% EXERCISE1: basic training and testing of a classifier

% setup MATLAB to use our software
setup ;

%% --------------------------------------------------------------------
% Stage A: Data Preparation
% --------------------------------------------------------------------

% Load training data
encoding = 'bovw' ;
%encoding = 'vlad' ;
%encoding = 'fv' ;

%category = 'motorbike' ;
%category = 'aeroplane' ;
%category = 'person' ;
regularization=[.1 1 3 10 30 100 300 1000];
colors={'b','r','g'};
j=1;
for category={'motorbike','aeroplane','person'}
APtrain=zeros(1,numel(regularization));
APtest=zeros(1,numel(regularization));
i=1;
for reg=regularization
pos = load(['data/' category{1} '_train_' encoding '.mat']) ;
neg = load(['data/background_train_' encoding '.mat']) ;

names = {pos.names{:}, neg.names{:}};
histograms = [pos.histograms, neg.histograms] ;
labels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
clear pos neg ;

% Load testing data
pos = load(['data/' category{1} '_val_' encoding '.mat']) ;
neg = load(['data/background_val_' encoding '.mat']) ;

testNames = {pos.names{:}, neg.names{:}};
testHistograms = [pos.histograms, neg.histograms] ;
testLabels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
clear pos neg ;

% For stage G: throw away part of the training data
%All the training set
%fraction = +inf ;

% sel = vl_colsubset(1:numel(labels), fraction, 'uniform') ;
% names = names(sel) ;
% histograms = histograms(:,sel) ;
% labels = labels(:,sel) ;
% clear sel ;

% count how many images are there
%fprintf('Number of training images: %d positive, %d negative\n', ...
        %sum(labels > 0), sum(labels < 0)) ;
%fprintf('Number of testing images: %d positive, %d negative\n', ...
        %sum(testLabels > 0), sum(testLabels < 0)) ;

% For Stage E: Vary the image representation
%histograms = removeSpatialInformation(histograms) ;
%testHistograms = removeSpatialInformation(testHistograms) ;

% For Stage F: Vary the classifier (Hellinger kernel)
%histograms=sqrt(histograms);
%testHistograms=sqrt(testHistograms);

% L2 normalize the histograms before running the linear SVM
histograms = bsxfun(@times, histograms, 1./sqrt(sum(histograms.^2,1))) ;
testHistograms = bsxfun(@times, testHistograms, 1./sqrt(sum(testHistograms.^2,1))) ;

% L1 normalize the histograms before running the linear SVM
%  histograms = bsxfun(@times, histograms, 1./(sum(abs(histograms),1))) ;
%  testHistograms = bsxfun(@times, testHistograms, 1./(sum(abs(testHistograms),1))) ;

% --------------------------------------------------------------------
% Stage B: Training a classifier
% --------------------------------------------------------------------

% Train the linear SVM. The SVM paramter C should be
% cross-validated. Here for simplicity we pick a value that works
% well with all kernels.
[w, bias] = trainLinearSVM(histograms, labels, reg) ;

% Evaluate the scores on the training data
scores = w' * histograms + bias ;

% Visualize the ranked list of images
%figure(1) ; clf ; set(1,'name','Ranked training images (subset)') ;
%displayRankedImageList(names, scores)  ;
%print(1,'-dpdf',sprintf('images/rank_train_%s.pdf',category), '-opengl')

%[~,best] = max(scores) ;
%displayRelevantVisualWords(names{best},w)

% Visualize the precision-recall curve
%figure(2) ; clf ; set(2,'name','Precision-recall on train data') ;
[~,~,info] = vl_pr(labels, scores) ;
APtrain(i)=info.auc;
clear info;

% --------------------------------------------------------------------
% Stage C: Classify the test images and assess the performance
% --------------------------------------------------------------------

% Test the linear SVM
testScores = w' * testHistograms + bias ;

% Visualize the ranked list of images
%figure(3) ; clf ; set(3,'name','Ranked test images (subset)') ;
%displayRankedImageList(testNames, testScores)  ;
%print(3,'-dpdf',sprintf('images/Hellinger_rank_test_%s.pdf',category), '-opengl')

% Visualize visual words by relevance on the first image
%[~,best] = max(testScores) ;
%figure(5) ; clf ; set(5,'name','Visual words most related to the class - aeroplane - ') ;
%displayRelevantVisualWords(testNames{best},w)
%print(5,'-dpdf',sprintf('images/vw_test_%s.pdf',category), '-opengl')

% Visualize the precision-recall curve
%figure(4) ; clf ; set(4,'name','Precision-recall on test data','PaperOrientation','landscape') ;
%vl_pr(testLabels, testScores) ;
%print(4,'-dpdf',sprintf('images/Hellinger_pr_test_%s.pdf',category), '-opengl')

% Print results
[~,~,info] = vl_pr(testLabels, testScores) ;
APtest(i)=info.auc;
i=i+1;
%fprintf('Test AP (category %s - fraction %.2f): %.2f\n', category{1},fraction,info.auc) ;
end
figure,
plot(regularization,APtrain,'color',colors{j},'marker','x','LineWidth',2);
hold on 
plot(regularization,APtest,'color',colors{j},'marker','x','LineWidth',2,'LineStyle','--');
xlabel('Regularization parameter C');
ylabel('mAP');
legend(sprintf('%s train',category{1}),sprintf('%s test',category{1}),'Location','southeast')
title(sprintf('C tuning - %s',category{1}))
hold off,
j=j+1;
end
%[~,perm] = sort(testScores,'descend') ;
%fprintf('Correctly retrieved in the top 36: %d\n', sum(testLabels(perm(1:36)) > 0)) ;
