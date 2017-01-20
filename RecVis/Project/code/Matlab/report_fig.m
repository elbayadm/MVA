%% Settings
clear; clc; close all;
warning 'off'
addpath(genpath('Matlab/'))
addpath(genpath('logs/'))
addpath(genpath('features/'))
addpath(genpath('data/'))
addpath('data/morpho/original')
addpath('~/GitHub/Morpho/OldClassifier')
set(0, 'DefaultAxesFontName', 'Helvetica')
set(0,'defaultTextFontName', 'Helvetica')
set(0,'DefaultAxesFontSize',12)
set(0,'DefaultTextFontSize',12)
%% Set colors
rand('seed',178);
col=[161 212 144;%jade
    252 215 5%Gold
    50 155 247;%bleu ciel
    9 125 18;%flash green
    2 2 214;%dark blue
    145 145 145;%Gris
    199 2 45; %brickred
    209 132 188;%Dusty pink
    237 99 7;%Dark orange
]/255;
%% Benchmarking:
B37_yes=importdata('B37_scoreWithGlasses.txt');
B37_yes=B37_yes.data;
B53_yes=importdata('B53_scoreWithGlasses.txt');
B53_yes=B53_yes.data;
B61_yes=importdata('B61_scoreWithGlasses.txt');
B61_yes=B61_yes.data;
B64_yes=importdata('B64_scoreWithGlasses.txt');
B64_yes=B64_yes.data;
B74_yes=importdata('B74_scoreWithGlasses.txt');
B74_yes=B74_yes.data;
B37_no=importdata('B37_scoreWithoutGlasses.txt');
B37_no=B37_no.data;
B53_no=importdata('B53_scoreWithoutGlasses.txt');
B53_no=B53_no.data;
B61_no=importdata('B61_scoreWithoutGlasses.txt');
B61_no=B61_no.data;
B64_no=importdata('B64_scoreWithoutGlasses.txt');
B64_no=B64_no.data;
B74_no=importdata('B74_scoreWithoutGlasses.txt');
B74_no=B74_no.data;
%%
%TR1
tr1_scores=importdata('Tr1_scores.mat');
tr1=importdata('subtrain_images.txt');
tr1=tr1.data;
tr1_yes=tr1_scores(tr1==1);
tr1_no=tr1_scores(tr1==0);

%TEST
te=importdata('test_os.txt');
te_scores=te.data(:,2);
te=te.data(:,1);
te_yes=te_scores(te==1);
te_no=te_scores(te==0);
%% Hists
xbins=linspace(0,250,100);
close all
figure(1); clf;
set(1,'units','normalized','position',[.1 .1 .7 .18])
set(1,'PaperType','A4','PaperPositionMode','auto','paperOrientation','landscape')

subplot(1,3,1)
h1=histogram(B53_yes,xbins);
set(h1,'FaceColor',col(7,:),'facealpha',.75);
hold on,
h2=histogram(B53_no,xbins);
set(h2,'FaceColor',col(1,:),'facealpha',.75);
xlabel('score')
ylabel('log(count)')
title('Benchmark set 1')
set(gca, 'YScale', 'log')
xlim([0 250])

subplot(1,3,2)
h1=histogram(B61_yes,xbins);
set(h1,'FaceColor',col(7,:),'facealpha',.75);
hold on,
h2=histogram(B61_no,xbins);
set(h2,'FaceColor',col(1,:),'facealpha',.75);
xlabel('score')
title('Benchmark set 2')
set(gca, 'YScale', 'log')
legend({'with glasses (1)', 'without glasses (0)'},'fontsize',10)
xlim([0 250])

subplot(1,3,3)
h1=histogram(B64_yes,xbins);
set(h1,'FaceColor',col(7,:),'facealpha',.75);
hold on,
h2=histogram(B64_no,xbins);
set(h2,'FaceColor',col(1,:),'facealpha',.75);
xlabel('score')
title('Benchmark set 3')
set(gca, 'YScale', 'log')
xlim([0 250])

print 'figures/rapport/tocrop/benchmark' '-dpdf'
%% Hists Tr1/Tr2/Test
xbins=linspace(0,250,100);
close all
figure(1); clf;
set(1,'units','normalized','position',[.1 .1 .5 .18])
set(1,'PaperType','A4','PaperPositionMode','auto','paperOrientation','landscape')

subplot(1,2,1)
h1=histogram(tr1_yes,xbins);
set(h1,'FaceColor',col(7,:),'facealpha',.75);
hold on,
h2=histogram(tr1_no,xbins);
set(h2,'FaceColor',col(1,:),'facealpha',.75);
xlabel('score')
ylabel('log(count)')
title('Tr1')
set(gca, 'YScale', 'log')
xlim([0 250])

subplot(1,2,2)
h1=histogram(te_yes,xbins);
set(h1,'FaceColor',col(7,:),'facealpha',.75);
hold on,
h2=histogram(te_no,xbins);
set(h2,'FaceColor',col(1,:),'facealpha',.75);
xlabel('score')
title('Test')
set(gca, 'YScale', 'log')
legend({'with glasses (1)', 'without glasses (0)'},'fontsize',10)
xlim([0 250])

print 'figures/rapport/tocrop/benchmark_trte' '-dpdf'
%% Old roc
close all, clc;
test_labels=importdata('test_labels.txt');
test_labels=2*test_labels.data-1;
test_scores=importdata('test_scores.txt');
[PREC, TPR, FPR, THRESH] = prec_rec(test_scores,test_labels,'numThresh',500,'plotROC',1,'plotPR',0,'col','r');
%print 'figures/rapport/tocrop/roc' '-dpdf'

prec_rec(test_scores,test_labels,'numThresh',500,'plotROC',0,'plotPR',1,'col','r');

[~,~,info]=vl_pr(test_labels,test_scores);
disp('ap'); info.ap
[tpr,tnr,info]=vl_roc(test_labels,test_scores);
disp('auc'); info.auc
hold on

%print 'figures/rapport/tocrop/pr' '-dpdf'
%% Threshold
figure(1), clf;
plot(THRESH,PREC,'color',col(5,:),'linewidth',2)
hold on
plot(THRESH,TPR,'color',col(7,:),'linewidth',2)
legend({'precision','recall'},'location','east')
xlabel('Threshold')
xlim([min(THRESH),max(THRESH)])
title('Precision and recall curves')
print 'figures/rapport/tocrop/pr_th' '-dpdf'
%% Clean databases:
close all;
%set(0,'DefaultAxesFontSize',14)
%set(0,'DefaultTextFontSize',14)
C1=importdata('morpho/NL000/prob1_10000.txt');
C2=importdata('morpho/NL000/prob2_10000.txt');
C3=importdata('morpho/NL000/Hprob_10000.txt');
figure,
[scores,pred]=max(C1,[],2);
pred=2*pred-3;
scores=scores.*pred;
[prec, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(5,:)); 
[~,~,info]=vl_pr(labels,scores);
disp('ap'); info.ap
[tpr,tnr,info]=vl_roc(labels,scores);
disp('auc'); info.auc
hold on
[scores,pred]=max(C2,[],2);
pred=2*pred-3;
scores=scores.*pred;
[prec, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(7,:));
[~,~,info]=vl_pr(labels,scores);
disp('ap'); info.ap
[tpr,tnr,info]=vl_roc(labels,scores);
disp('auc'); info.auc
hold on,
[scores,pred]=max(C3,[],2);
pred=2*pred-3;
scores=scores.*pred;
[prec, tpr, fpr, ~]=prec_rec(scores,labels,...
    'numThresh',500);
semilogx([0; fpr], [0 ; tpr], 'linewidth',2,'color',col(4,:)); 
[~,~,info]=vl_pr(labels,scores);
disp('ap'); info.ap
[tpr,tnr,info]=vl_roc(labels,scores);
disp('auc'); info.auc
legend({'Tr2','Tr1','Tr1- unbalanced cost'},'location','southeast');
xlabel('log(false positive rate)');
ylabel('true positive rate');
title('ROC curve');
%print 'figures/rapport/tocrop/clean_roc_log' '-dpdf'
%%
figure,
[scores,pred]=max(C1,[],2);
pred=2*pred-3;
scores=scores.*pred;
[prec, tpr, fpr, ~]=prec_rec(scores,test_labels,...
    'numThresh',500);
plot([0; tpr], [1; prec], 'linewidth',2,'color',col(5,:)); 
hold on
[scores,pred]=max(C2,[],2);
pred=2*pred-3;
scores=scores.*pred;
[prec, tpr, fpr, ~]=prec_rec(scores,test_labels,...
    'numThresh',500);
plot([0; tpr], [1; prec], 'linewidth',2,'color',col(7,:));

hold on,
[scores,pred]=max(C3,[],2);
pred=2*pred-3;
scores=scores.*pred;
[prec, tpr, fpr, ~]=prec_rec(scores,test_labels,...
    'numThresh',500);
plot([0; tpr], [1 ; prec], 'linewidth',2,'color',col(4,:));

legend({'Tr2','Tr1','Tr1- unbalanced cost'},'location','southeast');
xlabel('recall');
ylabel('precision'); 
title('precison-recall curve');
print 'figures/rapport/tocrop/clean_pr' '-dpdf'
%% Q matrix
% Get auc with conf.m
figure(1); clf;
set(1,'units','normalized','position',[.1 .1 .5 .4])
set(1,'PaperType','A4','PaperPositionMode','auto','paperOrientation','landscape')

subplot(2,1,1)
q0=[.01 .02 .03 .1 .15 .2];
q1=[.19 .38 .57 .01 .015 .02];
plot([0 q0],auc,'-o','color',col(5,:),'linewidth',2);
hold on,
plot(q0,auc_Q,'-o','color',col(7,:),'linewidth',2);
xlabel('Noise level on class 0')
ylabel('AUC')
legend({'baseline','with Q'})

subplot(2,1,2)
plot([0 q1(4:6) q1(1:3)],[auc(1) auc(5:7) auc(2:4)],'-o','color',col(5,:),'linewidth',2);
hold on,
plot([q1(4:6) q1(1:3)],[auc_Q(4:6) auc_Q(1:3)],'-o','color',col(7,:),'linewidth',2);
xlabel('Noise level on class 1')
ylabel('AUC')
legend({'baseline','with Q'})
print 'figures/rapport/tocrop/Qq' '-dpdf'

