clear; clc; close all;
warning 'off'
set(0, 'DefaultAxesFontName', 'Helvetica')
set(0,'defaultTextFontName', 'Helvetica')
set(0,'DefaultAxesFontSize',12)
set(0,'DefaultTextFontSize',12)
addpath(genpath('Matlab/'))
addpath(genpath('data/'))
addpath(genpath('features/'))
addpath(genpath('figures/'))
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
labels = importdata('test.txt');
labels = 2*labels.data-1;
%% Unbalanced costs
close all;
%set(0,'DefaultAxesFontSize',14)
%set(0,'DefaultTextFontSize',14)
fd ='features/infogain/';
Softmax = importdata('Tr1.txt');
C16=importdata([fd 'cprob16.txt']);
C15=importdata([fd 'cprob15.txt']);
C9=importdata([fd 'cprob9.txt']);
C17=importdata([fd 'cprob17.txt']);

i=1;
leg={};

figure(1); clf;
set(1,'units','normalized','position',[.1 .1 .44 .44])
set(1,'PaperType','A4','PaperPositionMode','auto','paperOrientation','landscape')

[scores,pred]=max(Softmax,[],2);
pred=2*pred-3;
scores=scores.*pred;
[~, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
%plot([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
[~,~,info1]=vl_pr(labels,scores);
[~,~,info2]=vl_roc(labels,scores);
leg{end+1} = sprintf('%7s | ap = %.3f | auc = %.3f','softmax',info1.ap,info2.auc);
hold on,
i=i+1;


[scores,pred]=max(C16,[],2);
pred=2*pred-3;
scores=scores.*pred;
[~, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
%plot([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
[~,~,info1]=vl_pr(labels,scores);
[~,~,info2]=vl_roc(labels,scores);
leg{end+1} = sprintf('%10s | ap = %.3f | auc = %.3f','H1',info1.ap,info2.auc);
hold on,
i=i+1;

[scores,pred]=max(C15,[],2);
pred=2*pred-3;
scores=scores.*pred;
[~, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
%plot([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
[~,~,info1]=vl_pr(labels,scores);
[~,~,info2]=vl_roc(labels,scores);
leg{end+1} = sprintf('%10s | ap = %.3f | auc = %.3f','H2',info1.ap,info2.auc);
hold on,
i=i+1;

[scores,pred]=max(C9,[],2);
pred=2*pred-3;
scores=scores.*pred;
[~, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:));
%plot([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
[~,~,info1]=vl_pr(labels,scores);
[~,~,info2]=vl_roc(labels,scores);
leg{end+1} = sprintf('%10s | ap = %.3f | auc = %.3f','H3',info1.ap,info2.auc);
hold on,
i=i+1;

[scores,pred]=max(C17,[],2);
pred=2*pred-3;
scores=scores.*pred;
[~, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:));
%plot([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
[~,~,info1]=vl_pr(labels,scores);
[~,~,info2]=vl_roc(labels,scores);
leg{end+1} = sprintf('%10s | ap = %.3f | auc = %.3f','H17',info1.ap,info2.auc);
hold on,
i=i+1;


legend(leg,'location','southeast'); 
xlabel('false positive rate');
ylabel('true positive rate');
title('ROC curve');
grid on,
box on, 

axis square,
% xlim([0 .2])
% ylim([.8 1])
%print 'figures/rapport/tocrop/unb1' '-dpdf'

%%  TR1 TR2 TR3 PerfTrx...
perfTr1=importdata([fd 'cprob9.txt']);
perfTr2=importdata('perfTr2.txt');
perfTr3=importdata('perfTr3.txt');
Tr1 = importdata('Tr1.txt');
Tr2=importdata('Tr2.txt');
Tr3=importdata('Tr3.txt');

i=1;
leg={};

figure(1); clf;
%set(1,'units','normalized','position',[.1 .1 .4 .4])
%set(1,'PaperType','A4','PaperPositionMode','auto','paperOrientation','landscape')

[scores,pred]=max(Tr1,[],2);
pred=2*pred-3;
scores=scores.*pred;
[prec, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
%semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:));
%plot([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
plot([0; tpr], [1; prec], 'linewidth',2,'color',col(i,:)); 
[~,~,info1]=vl_pr(labels,scores);
[~,~,info2]=vl_roc(labels,scores);
leg{end+1} = sprintf('Tr1 | ap = %.3f | auc = %.3f',info1.ap,info2.auc);
hold on,
i=i+1;

[scores,pred]=max(Tr2,[],2);
pred=2*pred-3;
scores=scores.*pred;
[prec, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
%semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:));
%plot([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
plot([0; tpr], [1; prec], 'linewidth',2,'color',col(i,:)); 
[~,~,info1]=vl_pr(labels,scores);
[~,~,info2]=vl_roc(labels,scores);
leg{end+1} = sprintf('Tr2 | ap = %.3f | auc = %.3f',info1.ap,info2.auc);
hold on,
i=i+1;

[scores,pred]=max(Tr3,[],2);
pred=2*pred-3;
scores=scores.*pred;
[prec, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
%semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:));
%plot([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
plot([0; tpr], [1; prec], 'linewidth',2,'color',col(i,:)); 
[~,~,info1]=vl_pr(labels,scores);
[~,~,info2]=vl_roc(labels,scores);
leg{end+1} = sprintf('Tr3 | ap = %.3f | auc = %.3f',info1.ap,info2.auc);
i=i+1;

[scores,pred]=max(perfTr1,[],2);
pred=2*pred-3;
scores=scores.*pred;
[prec, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
%semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
%plot([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
plot([0; tpr], [1; prec], 'linewidth',2,'color',col(i,:)); 
[~,~,info1]=vl_pr(labels,scores);
[~,~,info2]=vl_roc(labels,scores);
leg{end+1} = sprintf('perfTr1 | ap = %.3f | auc = %.3f',info1.ap,info2.auc);
hold on,
i=i+1;


[scores,pred]=max(perfTr2,[],2);
pred=2*pred-3;
scores=scores.*pred;
[prec, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
%semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
%plot([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
plot([0; tpr], [1; prec], 'linewidth',2,'color',col(i,:)); 
[~,~,info1]=vl_pr(labels,scores);
[~,~,info2]=vl_roc(labels,scores);
leg{end+1} = sprintf('perfTr2 | ap = %.3f | auc = %.3f',info1.ap,info2.auc);
hold on,
i=i+1;

[scores,pred]=max(perfTr3,[],2);
pred=2*pred-3;
scores=scores.*pred;
[prec, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
%semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
%plot([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(i,:)); 
plot([0; tpr], [1; prec], 'linewidth',2,'color',col(i,:)); 
[~,~,info1]=vl_pr(labels,scores);
[~,~,info2]=vl_roc(labels,scores);
leg{end+1} = sprintf('perfTr3 | ap = %.3f | auc = %.3f',info1.ap,info2.auc);
hold on,
i=i+1;


legend(leg,'location','southwest'); 
% xlabel('log(false positive rate)');
% ylabel('true positive rate');
xlabel('precision')
ylabel('recall')
title('Precision-recall curve');
grid on,
box on,
axis square
print 'figures/rapport/tocrop/perfXpr' '-dpdf'

%% Cost curves
loss =@(a,b,x) -(a*log(x)+b*log(1-x));
p=linspace(0,1,50);
leg={}; i=2;
figure(2), clf;

%H8
plot(p,loss(0,0.2,p),'color',col(i,:),'linewidth',1.5)
hold on, 
plot(p,loss(1,0,p),'-+','color',col(i,:),'linewidth',1.5)
hold on,
leg{end+1}='Negative H1';
leg{end+1}='Positive H1 = softmax';
i=i+1;

%H9
plot(p,loss(.01,.99,p),'color',col(i,:),'linewidth',1.5)
hold on, 
plot(p,loss(.6,.2,p),'-+','color',col(i,:),'linewidth',1.5)
leg{end+1}='Negative H2';
leg{end+1}='Positive H2';
i=i+1;
hold on,

%H13
plot(p,loss(.001,.95,p),'color',col(i,:),'linewidth',1.5)
hold on, 
plot(p,loss(.999,.05,p),'-+','color',col(i,:),'linewidth',1.5)
leg{end+1}='Negative H3';
leg{end+1}='Positive H3';
i=i+1;

legend(leg,'location','north','fontsize',13)
title('Cost functions','fontsize',16)
xlabel('P(y=1)','fontsize',16)
ylabel('cost','fontsize',16)
grid on

%print 'figures/rapport/tocrop/costs' '-dpdf'


%% Cost curves - slides
loss =@(a,b,x) -(a*log(x)+b*log(1-x));
p=linspace(0,1,50);
leg={}; i=1;
figure(2), clf;

%Softmax
plot(p,loss(0,1,p),'color','k','linewidth',1.5)
hold on, 
plot(p,loss(1,0,p),'-+','color','k','linewidth',1.5)
hold on,
leg{end+1}='negative';
leg{end+1}='softmax (+)';

%H16 = H1
hold on, 
plot(p,loss(.0526,.0,p),'-+','color',col(i,:),'linewidth',1.5)
leg{end+1}='H1 (+)';
i=i+2;

%H15 = H2
plot(p,loss(.2,.0,p),'-+','color',col(i,:),'linewidth',1.5)
leg{end+1}='H2(+)';
i=i+1;

%H13 = H3
plot(p,loss(.999,.05,p),'-+','color',col(i,:),'linewidth',1.5)
leg{end+1}='H3(+)';
hold on,
plot(p,loss(.001,.95,p),'color',col(i,:),'linewidth',1.5)
leg{end+1}='H3(-)';
i=i+1;

legend(leg,'location','north','fontsize',13)
title('Loss functions','fontsize',16)
xlabel('P(y=1)','fontsize',16)
ylabel('cost','fontsize',16)
grid on

print 'figures/rapport/tocrop/losses' '-dpdf'

