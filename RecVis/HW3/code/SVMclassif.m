%% --------------------------------------------------------------------
% Stage A: Data Preparation
% --------------------------------------------------------------------
close all

doprint=0;
category = 'motorbike' ;
%category = 'aeroplane' ;
%category = 'person' ;

encoding='cnn';

%Loading training data
pos = load(['data/' category '_train_' encoding '.mat']) ;
neg = load(['data/background_train_' encoding '.mat']) ;
names = [pos.names, neg.names];

%Loading validation data:
tpos = load(['data/' category '_val_' encoding '.mat']) ;
tneg = load(['data/background_val_' encoding '.mat']) ;
testNames= [tpos.names, tneg.names];
npos=size(pos.softmax,2);
nneg=size(neg.softmax,2);

tnpos=size(tpos.softmax,2);
tnneg=size(tneg.softmax,2);

labels = [ones(1,npos), - ones(1,nneg)];
testLabels = [ones(1,tnpos), - ones(1,tnneg)] ;

%Only the softmax output:
% select='_f1';
% features = [pos.softmax, neg.softmax] ;
% testFeatures = [tpos.softmax, tneg.softmax] ;

% softmax +fc8
% select='_f2';
% features = [[pos.softmax;pos.fc8], [neg.softmax;neg.fc8]] ;
% testFeatures = [[tpos.softmax;tpos.fc8], [tneg.softmax;tneg.fc8]] ;

% softmax +fc8 +fc7
select='_f3';
features = [[pos.softmax;pos.fc8;pos.fc7], [neg.softmax;neg.fc8;neg.fc7]] ;
testFeatures = [[tpos.softmax;tpos.fc8;tpos.fc7], [tneg.softmax;tneg.fc8;tneg.fc7]] ;

% count how many images are there
fprintf('Number of training images: %d positive, %d negative\n', ...
        sum(labels > 0), sum(labels < 0)) ;
fprintf('Number of testing images: %d positive, %d negative\n', ...
        sum(testLabels > 0), sum(testLabels < 0)) ;

    
%skip normalization
%normalization='';
    
%L2 normalize the histograms before running the linear SVM
% normalization='_l2';    
% features = bsxfun(@times, features, 1./sqrt(sum(features.^2,1))) ;
% testFeatures = bsxfun(@times, testFeatures, 1./sqrt(sum(testFeatures.^2,1))) ;

% L1 normalize the histograms before running the linear SVM
normalization='_l1'; 
features = bsxfun(@times, features, 1./(sum(abs(features),1))) ;
testFeatures = bsxfun(@times, testFeatures, 1./(sum(abs(testFeatures),1))) ;

%% --------------------------------------------------------------------
% Stage B: Training a classifier
% --------------------------------------------------------------------

% Train the linear SVM. The SVM paramter C should be
% cross-validated. Here for simplicity we pick a value that works
% well with all kernels.
regularization=[.1 1 3 10 30 100 300 1000];
APtrain=zeros(1,numel(regularization));
APtest=zeros(1,numel(regularization));
i=1;
for reg=regularization
    %Train
    [w, bias] = trainLinearSVM(features, labels, reg) ;
    scores = w' * features + bias ;
    %The precision-recall curve
    [~,~,info] = vl_pr(labels, scores) ;
    APtrain(i)=info.auc;
    clear info;
    
    % Test the linear SVM
    testScores = w' * testFeatures + bias ;
    [~,~,info] = vl_pr(testLabels, testScores) ;
    APtest(i)=info.auc;
    i=i+1;
end
figure(1),
plot(regularization,APtrain,'color','r','marker','x','LineWidth',2);
hold on 
plot(regularization,APtest,'color','r','marker','x','LineWidth',2,'LineStyle','--');
xlabel('Regularization parameter C');
ylabel('mAP');
legend(sprintf('%s train',category),sprintf('%s test',category),'Location','southeast')
[~,opt_C]=max(APtest);
title(sprintf('C tuning - %s\n Optimal C=%.3f',category,regularization(opt_C)),'fontsize',16)
hold off,
if doprint
print(1,'-dpdf',sprintf('HW3/images/tocrop/tuning_%s%s%s.pdf',category,normalization,select), '-opengl')
end
%%
%Tuned model
[w, bias] = trainLinearSVM(features, labels, opt_C) ;

figure(3) ; clf ; set(3,'name','Precision-recall on test data','PaperOrientation','landscape') ;
vl_pr(testLabels, testScores) ;
if doprint
print(3,'-dpdf',sprintf('HW3/images/tocrop/AP_%s%s%s.pdf',category,normalization,select), '-opengl')
end
%
figure(4) ; clf ; set(3,'name','Ranked test images (subset)') ;
displayRankedImageList(testNames, testScores)  ;
if doprint
print(4,'-dpdf',sprintf('HW3/images/tocrop/top_%s%s%s.pdf',category,normalization,select), '-opengl')
end
[~,perm]=sort(testScores,'descend');
fprintf('Correctly retrieved in the top 36: %d\n', sum(testLabels(perm(1:36)) > 0)) ;

%% Nnet Classes
doprint=0;
testFeatures_softmax=[tpos.softmax, tneg.softmax] ;
[bestScore, best] = sort(testFeatures_softmax,'descend') ;
class=net.classes.description(best(1:2,1:10));

if doprint
fileID = fopen(sprintf('HW3/top10_%s%s%s.txt',category,normalization,select),'w');
formatSpec = '%s & %s\\\\\n\\hline\n';
[nrows,ncols] = size(class);
for col = 1:ncols
    fprintf(fileID,formatSpec,class{:,col});
end
end

%% False positives and false negatives:
doprint=0;

pred=sign(testScores);

FP=find(testLabels~=pred & pred==1);
[~,perm]=sort(testScores(FP),'descend');
FP=FP(perm);
classFP=net.classes.description(best(1,FP));

FN=find(testLabels~=pred & pred==-1);
[~,perm]=sort(testScores(FN));
FN=FN(perm);
classFN=net.classes.description(best(1,FN));


figure(5),
sub=min(length(FP),6);
for i = 1:sub
  vl_tightsubplot(sub,i,'box','inner') ;
  fullPath = fullfile('data','images',[testNames{FP(i)} '.jpg']) ;
  imagesc(imread(fullPath)) ;
  text(10,10,sprintf('score: %.2f', testScores(FP(i))),...
       'background','w',...
       'verticalalignment','top', ...
       'fontsize', 8) ;
  yl=get(gca,'ylim');
  text(10,yl(2)-100,sprintf('NNet class:\n %s', strjoin(strsplit(classFP{i},','),'\n')),...
       'background','w',...
       'verticalalignment','top', ...
       'fontsize', 10) ;
  set(gca,'xtick',[],'ytick',[]) ; axis image ;
end

if doprint
print(5,'-dpdf',sprintf('HW3/images/tocrop/FP_%s%s%s.pdf',category,normalization,select), '-opengl')
end

figure(6),
sub=min(length(FN),6);
for i = 1:sub
  vl_tightsubplot(sub,i,'box','inner') ;
  fullPath = fullfile('data','images',[testNames{FN(i)} '.jpg']) ;
  imagesc(imread(fullPath)) ;
  text(10,10,sprintf('score: %.2f', testScores(FN(i))),...
       'background','w',...
       'verticalalignment','top', ...
       'fontsize', 8) ;
  yl=get(gca,'ylim');
  text(10,yl(2)-100,sprintf('NNet class:\n %s', strjoin(strsplit(classFN{i},','),'\n')),...
       'background','w',...
       'verticalalignment','top', ...
       'fontsize', 10) ;
  set(gca,'xtick',[],'ytick',[]) ; axis image ;
end

if doprint
print(6,'-dpdf',sprintf('HW3/images/tocrop/FN_%s%s%s.pdf',category,normalization,select), '-opengl')
end

Ctest = strsplit(classFN{i},',');
