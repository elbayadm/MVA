%% Settings
clearvars; close all; clc;
warning 'off'
addpath(genpath('Matlab/'))
addpath(genpath('logs/'))
addpath(genpath('features/'))
addpath('data/original')
%addpath(genpath('~/Github/toolbox'))
addpath(genpath('/usr/local/cellar/vlfeat'))
%%
rand('seed',178);
col=[
    9 125 18;%Green
    97 6 158;%Violet
    199 2 45; %brickred
    2 2 214;%dark blue
    237 99 7;%Dark orange
    145 145 145;%Gris
    127 245 163;%Light green
    252 215 5%Gold
    50 155 247;%bleu ciel
]/255;

%% Logs and features:
l30=importdata('isnoisy30.log.train');
l30t=importdata('isnoisy30.log.test');
l40=importdata('isnoisy40.log.train');
l40t=importdata('isnoisy40.log.test');
%% Accuracy and losses:
%Loss
figure(1), clf
set(1,'units','normalized','position',[.1 .1 .6 .6])
set(1,'PaperType','A4','PaperPositionMode','auto','PaperOrientation','landscape')
subplot(3,1,1)
plot(l30.data(:,1),l30.data(:,4),'color',col(1,:),'linewidth',1.5);
hold on,
plot(l40.data(:,1),l40.data(:,4),'color',col(2,:),'linewidth',1.5);

title('Training loss')
ylabel('Loss')
grid on
grid minor
legend({'th=30','th=40'},'location','northwest','fontsize',10)

subplot(3,1,2)
plot(l30t.data(:,1),l30t.data(:,5),'color',col(1,:),'linewidth',1.5);
hold on,
plot(l40t.data(:,1),l40t.data(:,5),'color',col(2,:),'linewidth',1.5);
title('Test loss')
ylabel('Loss')
grid on
grid minor
legend({'th=30','th=40'},'location','northwest','fontsize',10)


subplot(3,1,3)
plot(l30t.data(:,1),l30t.data(:,4),'color',col(1,:),'linewidth',1.5);
hold on,
plot(l40t.data(:,1),l40t.data(:,4),'color',col(2,:),'linewidth',1.5);
title('Accuracy')
xlabel('Iteration')
legend({'th=30','th=40'},'location','northwest','fontsize',10)
ylim([.93 1])
grid on
grid minor
print 'figures/isnoisy/tocrop/loss_acc' '-dpdf'
%% disp stats
clc;
th=40;
disp('Test')
labels=importdata('test.txt');
labels=labels.data;
scores=importdata('test_scores.txt');
scores=scores.data;
noisy_labels=scores>th;
isnoisy_gt=2*noisy_labels+labels;
fprintf('noisy=true=0 : %d\n',sum(isnoisy_gt==0));
fprintf('noisy=0 true=1 : %d\n',sum(isnoisy_gt==1));
fprintf('noisy=1 true=0 : %d\n',sum(isnoisy_gt==2));
fprintf('noisy=true=1 : %d\n',sum(isnoisy_gt==3));
disp('')
disp('Train')
labelsT=importdata('Tr1_labels.mat');
labelsT=double(labelsT');
scoresT=importdata('Tr1_scores.mat');

noisy_labelsT=scoresT>th;
isnoisyT=2*noisy_labelsT+labelsT;
fprintf('noisy=true=0 : %d\n',sum(isnoisyT==0));
fprintf('noisy=0 true=1 : %d\n',sum(isnoisyT==1));
fprintf('noisy=1 true=0 : %d\n',sum(isnoisyT==2));
fprintf('noisy=true=1 : %d\n',sum(isnoisyT==3));
%% check prediction:
th30=importdata('prob30.txt');
th40=importdata('prob40.txt');

[~,ntype30]=max(th30,[],2);
ntype30=ntype30-1;
disp('Test - prediction (SGD)')
fprintf('noisy=true=0 : %d\n',sum(ntype30==0));
fprintf('noisy=0 true=1 : %d\n',sum(ntype30==1));
fprintf('noisy=1 true=0 : %d\n',sum(ntype30==2));
fprintf('noisy=true=1 : %d\n',sum(ntype30==3));
%% One vs all Roc curves:
feat = th30;
[prob,ntype]=max(feat,[],2);
ntype=ntype-1;

% Class 0:
prob(ntype~=0)=1-feat(ntype~=0,1);
ntype=2*(ntype==0)-1;
score=prob.*ntype;
t_labels=2*(isnoisy==0)-1;
prec_rec(score,t_labels,'numThresh',8000,'plotROC',1,'plotPR',0,'col',col(1,:),'plotBaseline',1);

[prob,ntype]=max(feat,[],2);
ntype=ntype-1;
% Class 1:
prob(ntype~=1)=1-feat(ntype~=1,2);
ntype=2*(ntype==1)-1;
score=prob.*ntype;
t_labels=2*(isnoisy==1)-1;
prec_rec(score,t_labels,'numThresh',8000,'plotROC',1,'plotPR',0,'col',col(2,:),'holdFigure',1);

[prob,ntype]=max(feat,[],2);
ntype=ntype-1;
% Class 2:
prob(ntype~=2)=1-feat(ntype~=2,3);
ntype=2*(ntype==2)-1;
score=prob.*ntype;
t_labels=2*(isnoisy==2)-1;
prec_rec(score,t_labels,'numThresh',8000,'plotROC',1,'plotPR',0,'col',col(3,:),'holdFigure',1);

[prob,ntype]=max(feat,[],2);
ntype=ntype-1;
% Class 3:
prob(ntype~=3)=1-feat(ntype~=3,4);
ntype=2*(ntype==3)-1;
score=prob.*ntype;
t_labels=2*(isnoisy==3)-1;
prec_rec(score,t_labels,'numThresh',8000,'plotROC',1,'plotPR',0,'col',col(4,:),'holdFigure',1);
title('ROC curve - th = 30')
legend({'random','noisy=true=0','noisy=0 true=1','noisy=1 true=0','noisy=true=1'},'location','southeast')
print 'figures/isnoisy/tocrop/roc_multiclass_30' '-dpdf'
%% PR
feat1 = th30;
feat2 = th40;
[prob,ntype]=max(feat,[],2);
ntype=ntype-1;

[prob2,ntype]=max(feat2,[],2);
ntype=ntype-1;

% Class 0:
prob(ntype~=0)=1-feat(ntype~=0,1);
ntype=2*(ntype==0)-1;
score=prob.*ntype;

prob2(ntype~=0)=1-feat2(ntype~=0,1);
ntype=2*(ntype==0)-1;
score2=prob2.*ntype;

t_labels=2*(isnoisy==0)-1;
prec_rec(score,t_labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col',col(4,:),'plotBaseline',1);
prec_rec(score2,t_labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col',col(2,:),'holdFigure',1);
legend({'random','adagrad','sgd'},'location','southwest')
title('noisy = true = 0','fontsize',18)
axis square,
print 'figures/isnoisy/tocrop/pr_c0' '-dpdf'

[prob,ntype]=max(feat,[],2);
ntype=ntype-1;

[prob2,ntype]=max(feat2,[],2);
ntype=ntype-1;

% Class 1:
prob(ntype~=1)=1-feat(ntype~=1,2);
ntype=2*(ntype==1)-1;
score=prob.*ntype;

prob2(ntype~=1)=1-feat2(ntype~=1,2);
ntype=2*(ntype==1)-1;
score2=prob2.*ntype;


t_labels=2*(isnoisy==1)-1;
prec_rec(score,t_labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col',col(4,:));
prec_rec(score2,t_labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col',col(2,:),'holdFigure',1);
legend({'random','adagrad','sgd'},'location','southwest')
title('noisy =0  true = 1','fontsize',18)
axis square,
print 'figures/isnoisy/tocrop/pr_c1' '-dpdf'

[prob,ntype]=max(feat,[],2);
ntype=ntype-1;

[prob2,ntype]=max(feat2,[],2);
ntype=ntype-1;

% Class 2:
prob(ntype~=2)=1-feat(ntype~=2,3);
ntype=2*(ntype==2)-1;
score=prob.*ntype;

prob2(ntype~=2)=1-feat2(ntype~=2,3);
ntype=2*(ntype==2)-1;
score2=prob2.*ntype;

t_labels=2*(isnoisy==2)-1;
prec_rec(score,t_labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col',col(4,:));
prec_rec(score2,t_labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col',col(2,:),'holdFigure',1);
legend({'random','adagrad','sgd'},'location','southwest')
title('noisy =1  true = 0','fontsize',18)
axis square,
print 'figures/isnoisy/tocrop/pr_c2' '-dpdf'

[prob,ntype]=max(feat,[],2);
ntype=ntype-1;

[prob2,ntype]=max(feat2,[],2);
ntype=ntype-1;

% Class 3:
prob(ntype~=3)=1-feat(ntype~=3,4);
ntype=2*(ntype==3)-1;
score=prob.*ntype;

prob2(ntype~=3)=1-feat2(ntype~=3,4);
ntype=2*(ntype==3)-1;
score2=prob2.*ntype;

t_labels=2*(isnoisy==3)-1;
prec_rec(score,t_labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col',col(4,:));
prec_rec(score2,t_labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col',col(2,:),'holdFigure',1);
legend({'random','adagrad','sgd'},'location','southwest')
title('noisy = true = 1','fontsize',18)
axis square,
print 'figures/isnoisy/tocrop/pr_c3' '-dpdf'
%% Scores distrib:
xbins=linspace(0,250,100);
close all
figure(1); clf;
set(1,'units','normalized','position',[.1 .1 .5 .5])
set(1,'PaperType','A4','PaperPositionMode','auto','paperOrientation','landscape')

h1=histogram(scoresT(isnoisyT==0),xbins);
set(h1,'FaceColor',col(1,:),'facealpha',.75);
hold on,

h2=histogram(scoresT(isnoisyT==1),xbins);
set(h2,'FaceColor',col(2,:),'facealpha',.75);
hold on,

h3=histogram(scoresT(isnoisyT==2),xbins);
set(h2,'FaceColor',col(3,:),'facealpha',.75);
hold on,

h4=histogram(scoresT(isnoisyT==3),xbins);
set(h4,'FaceColor',col(4,:),'facealpha',.75);

xlabel('score')
ylabel('count')
title('Scores')
set(gca, 'YScale', 'log')
xlim([0 250])
%% Relabel the full training set - Output file

th = 40;
Feat = importdata('probTF40.txt');
noisy_labels = importdata('full_train_scores.txt');
noisy_labels = noisy_labels.data;
noisy_labels = noisy_labels > th;
[~,isnoisy] = max(Feat,[],2);
isnoisy = isnoisy -1;
sum(isnoisy==0)
sum(isnoisy==1)
sum(isnoisy==2)
sum(isnoisy==3)
cleaned_labels = noisy_labels;
cleaned_labels(isnoisy== 0)=0;
cleaned_labels(isnoisy== 1)= 1;
cleaned_labels(isnoisy== 2)= 0;
cleaned_labels(isnoisy== 3)= 1;
out = fopen('cleaned30.txt','w');
fprintf(out,'%d\n',cleaned_labels);
disp('Done')
%% Check on Test: 
true_labels = importdata('test.txt');
true_labels = true_labels.data;

th = 30;
Feat = importdata('prob40.txt');
noisy_labels = importdata('test_scores.txt');
noisy_labels = noisy_labels.data;
old_scores = noisy_labels;
noisy_labels = noisy_labels > th;
[~,isnoisy] = max(Feat,[],2);
isnoisy = isnoisy -1;
sum(isnoisy==0)
sum(isnoisy==1)
sum(isnoisy==2)
sum(isnoisy==3)
cleaned_labels = noisy_labels;
cleaned_labels(isnoisy== 0)=0;
cleaned_labels(isnoisy== 1)= 1;
cleaned_labels(isnoisy== 2)= 0;
cleaned_labels(isnoisy== 3)= 1;

% Accuracy
fprintf('Accuracy : %.3f\n',sum(cleaned_labels==true_labels)/length(true_labels));
%% Visualize:
%% Plot some samples
testIM = importdata('test.mat');
%% 
I = find((cleaned_labels~=noisy_labels).*(cleaned_labels==true_labels).*(true_labels==0));
figure(1) ; clf ;                                              %[.1 .1 .3 .42]
%set(1,'name','False positives','units','normalized','position',[.1 .1 .3 .42])
%set(1,'PaperType','A4','PaperPositionMode','auto') ;
n=min(length(I),6);
  for i =(1:n)
    vl_tightsubplot(n,i,'box','inner') ;
    imagesc(squeeze(testIM(I(i),:,:,:))) ;
    text(5,80,sprintf('c_%d | %d?%d',isnoisy(I(i)), noisy_labels(I(i)),cleaned_labels(I(i))),...
       'color','w',...
       'background','k',...
       'Margin',.1,...
       'verticalalignment','top', ...
       'fontsize', 12);
   text(50,5,sprintf('old = %d',round(old_scores(I(i)))),...
       'color','k',...
       'background','w',...
       'Margin',.1,...
       'verticalalignment','top', ...
       'fontsize', 10);
    set(gca,'xtick',[],'ytick',[]) ; axis image ;
  end
print 'figures/isnoisy/tocrop/flip_neg_th30' '-dpdf'
%% Confusion matrix
figure(1) ; clf ;                                             
set(1,'name','False positives','units','normalized','position',[.1 .1 .2 .3])
set(1,'PaperType','A4','PaperPositionMode','auto') ;
cfmat(isnoisy,isnoisy_gt,0:3);
print 'figures/isnoisy/tocrop/cfmat40' '-dpdf'
%% The cleaned model:
set(0,'DefaultAxesFontSize',13)
set(0,'DefaultTextFontSize',14)

labels = importdata('test.txt');
labels = 2*labels.data-1;

pred30 = importdata('pred30.txt');
pred40 = importdata('pred40.txt');
Softmax = importdata('th30.txt');
Softmax40 = importdata('th40.txt');

perfTr3 = importdata('perfTr3.txt');
Tr3strap1 = importdata('Tr3strap1.txt');
i=1;
leg={};

figure(1); clf;
set(1,'units','normalized','position',[.1 .1 .6 .6])
set(1,'PaperType','A4','PaperPositionMode','auto','paperOrientation','landscape')

% [scores,pred]=max(Softmax,[],2);
% pred=2*pred-3;
% scores=scores.*pred;
% [~, tpr, fpr, ~]=prec_rec(scores,labels,...
%     'numThresh',500);
% semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
% %plot([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
% [~,~,info1]=vl_pr(labels,scores);
% [~,~,info2]=vl_roc(labels,scores);
% leg{end+1} = sprintf('%s | ap = %.3f | auc = %.3f','   original th30',info1.ap,info2.auc);
% hold on,
% i=i+1;
% 
% 
% [scores,pred]=max(pred30,[],2);
% pred=2*pred-3;
% scores=scores.*pred;
% [~, tpr, fpr, ~]=prec_rec(scores,labels,...
%     'numThresh',500);
% semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
% %plot([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
% [~,~,info1]=vl_pr(labels,scores);
% [~,~,info2]=vl_roc(labels,scores);
% leg{end+1} = sprintf('%s | ap = %.3f | auc = %.3f','cleansed th30',info1.ap,info2.auc);
% 
% hold on,
% i=i+1;

[scores,pred]=max(Softmax40,[],2);
pred=2*pred-3;
scores=scores.*pred;
[~, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
%plot([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
[~,~,info1]=vl_pr(labels,scores);
[~,~,info2]=vl_roc(labels,scores);
leg{end+1} = sprintf('%s | ap = %.3f | auc = %.3f','   original th40',info1.ap,info2.auc);
hold on,
i=i+1;

[scores,pred]=max(pred40,[],2);
pred=2*pred-3;
scores=scores.*pred;
[~, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
%plot([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
[~,~,info1]=vl_pr(labels,scores);
[~,~,info2]=vl_roc(labels,scores);
leg{end+1} = sprintf('%s | ap = %.3f | auc = %.3f','cleansed th40',info1.ap,info2.auc);

hold on,
i=i+1;

[scores,pred]=max(Tr3strap1,[],2);
pred=2*pred-3;
scores=scores.*pred;
[~, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
%plot([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
[~,~,info1]=vl_pr(labels,scores);
[~,~,info2]=vl_roc(labels,scores);
leg{end+1} = sprintf('%s | ap = %.3f | auc = %.3f','it=1 extension',info1.ap,info2.auc);

hold on,
i=i+1;

[scores,pred]=max(perfTr3,[],2);
pred=2*pred-3;
scores=scores.*pred;
[~, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
%plot([0; fpr], [0 ; tpr], 'linewidth',1.7,'color','k','linestyle','--'); 
[~,~,info1]=vl_pr(labels,scores);
[~,~,info2]=vl_roc(labels,scores);
leg{end+1} = sprintf('%s | ap = %.3f | auc = %.3f','           perfTr3',info1.ap,info2.auc);


legend(leg,'location','southeast','fontsize',10); 
xlabel('false positive rate');  
ylabel('true positive rate');
title('ROC curve - cleansing');
grid on,
box on, 

axis square,
% xlim([0.4 1])
% ylim([.9 1])
print 'figures/isnoisy/tocrop/pred_log' '-dpdf'
%%
