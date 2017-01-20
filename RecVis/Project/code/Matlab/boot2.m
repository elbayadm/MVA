%% Settings
clear; clc; close all;
warning 'off'
addpath(genpath('Matlab/'))
addpath(genpath('logs/'))
addpath(genpath('features/'))
addpath('Confusion/')
addpath('data/morpho_QQ/bs2')
addpath(genpath('data/morpho_QQ/glasses'))
set(0, 'DefaultAxesFontName', 'Helvetica')
set(0,'defaultTextFontName', 'Helvetica')
set(0,'DefaultAxesFontSize',10)
set(0,'DefaultTextFontSize',10)
%% Set colors
rand('seed',178);
col=[161 212 144;%jade
    252 215 5%Gold
    50 155 247;%bleu ciel
    237 99 7;%Dark orange
    2 2 214;%dark blue
    145 145 145;%Gris
    199 2 45; %brickred
    209 132 188;%Dusty pink
    9 125 18;%flash green
    ]/255;
labels=importdata('test_labels.txt');
labels=2*labels.data-1;
%% Check colors
close all;
figure,
for i=1:size(col,1)
    plot(1:10,i*ones(1,10),'color',col(i,:),'linewidth',3)
    hold on
end
%% Features & logs:
% logs:
b5=importdata('morpho/NLTH30_30/p1.log.test');
b5T=importdata('morpho/NLTH30_30/p1.log.train');
b20=importdata('morpho/NLTH30_30/bp1.log.test');
b20T=importdata('morpho/NLTH30_30/bp1.log.train');
clean_log=importdata('morpho/NL000/clean1.log.test');

s1=importdata('morpho/NLTH30_30/bs1/strap1.log.test');
s1T=importdata('morpho/NLTH30_30/bs1/strap1.log.train');
bs2=importdata('morpho/NLTH30_30/bs2/strap2.log.test');
bs2T=importdata('morpho/NLTH30_30/bs2/strap2.log.train');
bs3=importdata('morpho/NLTH30_30/bs2/strap3.log.test');
bs3T=importdata('morpho/NLTH30_30/bs2/strap3.log.train');
bs4=importdata('morpho/NLTH30_30/bs2/strap4.log.test');
bs4T=importdata('morpho/NLTH30_30/bs2/strap4.log.train');
bs5=importdata('morpho/NLTH30_30/bs2/strap5.log.test');
bs5T=importdata('morpho/NLTH30_30/bs2/strap5.log.train');
bs6=importdata('morpho/NLTH30_30/bs2/strap6.log.test');
bs6T=importdata('morpho/NLTH30_30/bs2/strap6.log.train');
%bs6_crop=importdata('morpho/NLTH30_30/bs2/strap6_crop.log.test');
%bs6T_crop=importdata('morpho/NLTH30_30/bs2/strap6_crop.log.train');

% Features:
% train:
feat20T=importdata('NLTH30_30/train_bprob_10000.txt');
feat5T=importdata('NLTH30_30/train_prob_10000.txt');
strap1T=importdata('NLTH30_30/bs1/train_strap1_10000.txt');
bstrap2T=importdata('NLTH30_30/bs2/train_strap2_10000.txt');
bstrap3T=importdata('NLTH30_30/bs2/train_strap3_10000.txt');
bstrap4T=importdata('NLTH30_30/bs2/train_strap4_10000.txt');
bstrap5T=importdata('NLTH30_30/bs2/train_strap5_10000.txt');
bstrap6T=importdata('NLTH30_30/bs2/train_strap6_10000.txt');

% test:
feat20=importdata('NLTH30_30/bprob_10000.txt');
feat5=importdata('NLTH30_30/prob_10000.txt');
clean=importdata('morpho/NL000/prob1_10000.txt');

strap1=importdata('NLTH30_30/bs1/test_strap1_10000.txt');
bstrap2=importdata('NLTH30_30/bs2/test_strap2_10000.txt');
bstrap3=importdata('NLTH30_30/bs2/test_strap3_10000.txt');
bstrap4=importdata('NLTH30_30/bs2/test_strap4_10000.txt');
bstrap5=importdata('NLTH30_30/bs2/test_strap5_10000.txt');
bstrap6=importdata('NLTH30_30/bs2/test_strap6_10000.txt');
%bstrap6_crop=importdata('NLTH30_30/bs2/test_strap6_crop_10000.txt');



% labels
list20=importdata('balanced_labels_30_30.txt');
list5=importdata('labels_30_30.txt');
labelsTT=importdata('out_glasses_lfw.txt');
labelsTT=2*labelsTT.data-1;
ls1=importdata('strap1_balanced_labels_30_30.txt');
bls2=importdata('bstrap2_balanced_labels_30_30.txt');
bls3=importdata('bstrap3_balanced_labels_30_30.txt');
bls4=importdata('bstrap4_balanced_labels_30_30.txt');
bls5=importdata('bstrap5_balanced_labels_30_30.txt');
bls6=importdata('bstrap6_balanced_labels_30_30.txt');



% Bootstrapping scores:
s1scores=importdata('strap1_balanced_scores_30_30.txt');
bs2scores=importdata('bstrap2_balanced_scores_30_30.txt');
bs3scores=importdata('bstrap3_balanced_scores_30_30.txt');
bs4scores=importdata('bstrap4_balanced_scores_30_30.txt');
bs5scores=importdata('bstrap5_balanced_scores_30_30.txt');
bs6scores=importdata('bstrap6_balanced_scores_30_30.txt');

% Old scores:
full_train=importdata('full_train.txt');
full_train_scores=full_train.data;
full_train_names=full_train.textdata;

%Train subset:
train=importdata('train_os.txt');
train_names=train.textdata;
train_labels=train.data(:,1);
train_scores=train.data(:,2);
clear train
%% Misclassified on the training 
I=find(list20~=-1);
names=full_train_names(I);
old_scores=[full_train_scores(I), s1scores(I), bs2scores(I),bs3scores(I),bs4scores(I),bs5scores(I),bs6scores(I)];

% Training labels
true_labels=2*bls6(I)-1;

% Model scores
[scores,pred]=max(bstrap6T(I,:),[],2);
pred=2*pred-3;
scores=scores.*pred;

FP=find((true_labels==-1).*(pred==1));
FN=find((true_labels==1).*(pred==-1));
length(FP)
length(FN)
%% Plot some samples (FP):
close all;
figure(1) ; clf ; 
set(1,'name','False positives','units','normalized','position',[.1 .1 .4 .6])
set(1,'PaperType','A4','PaperPositionMode','auto') ;
n=121;
  for i =(1:n)
    vl_tightsubplot(n,i,'box','inner') ;
    if exist(char(names(FP(i))), 'file')
      fullPath = char(names(FP(i))) ;
    else
      fprintf(2,'Cannot find file %s',char(names(FP(i)))); 
    end
    imagesc(imread(fullPath)) ;
    text(10,10,sprintf('%d', FP(i)),...
       'color','w',...
       'background','k',...
       'Margin',.1,...
       'verticalalignment','top', ...
       'fontsize', 2) ;
    text(50,70,sprintf('%.2f//%d/%d/\n%d/%d/%d/%d/%d', scores(FP(i)),round(old_scores(FP(i),:))),...
       'background','w',...
       'verticalalignment','top', ...
       'horizontalalignment','center', ...
       'Margin',.1,...
       'fontsize', 5) ;
    set(gca,'xtick',[],'ytick',[]) ; axis image ;
  end
print 'figures/balanced/FNFP/FP_bs6' '-dpdf'
%% Plot some samples (FN):
figure(2) ; clf ; set(2,'name','False negatives','units','normalized','position',[.1 .1 .4 .6]);
set(2,'PaperType','A4','PaperPositionMode','auto');
  for i =(1:n)
    vl_tightsubplot(n,i,'box','inner') ;
    if exist(char(names(FN(i))), 'file')
      fullPath = char(names(FN(i))) ;
    else
      fprintf(2,'Cannot find file %s',char(names(FN(i)))); 
    end
    imagesc(imread(fullPath));
     text(10,10,sprintf('%d', FN(i)),...
       'color','w',...
       'background','k',...
       'Margin',.1,...
       'verticalalignment','top', ...
       'fontsize', 2) ;
    text(50,70,sprintf('%.2f//%d/%d/\n%d/%d/%d/%d/%d', scores(FN(i)),round(old_scores(FN(i),:))),...
       'background','w',...
       'verticalalignment','top', ...
       'horizontalalignment','center', ...
       'Margin',.1,...
       'fontsize', 5) ;
    set(gca,'xtick',[],'ytick',[]) ; axis image ;
  end
  print 'figures/balanced/FNFP/FN_bs6' '-dpdf'  
%% Bootstrapping 20%: - Build the new training labels
I=find(list20~=-1);
hist_scores=[bs5scores(I),100*bstrap5T(I,2)];
new_scores=mean(hist_scores,2);
% figure,
% subplot(3,1,1)
% hist(old_scores,100);
% subplot(3,1,2)
% hist(cnn_scores,100);
% subplot(3,1,3)
% hist(new_scores,100);
th=50; 
new_labels=new_scores>th;
new_list=list20;
new_list(I)=new_labels;
sum(new_labels==1)/length(new_labels)
sum(new_labels==0)/length(new_labels)

% Write them down:
% Scores
full_new_scores=-ones(size(full_train_scores));
full_new_scores(I)=new_scores;
file_ = fopen('data/morpho_QQ/bstrap6_balanced_scores_30_30.txt','wt');
fprintf(file_,'%.3f\n',full_new_scores);
fclose(file_);
disp('Done')
% Labels
file_ = fopen('data/morpho_QQ/bstrap6_balanced_labels_30_30.txt','wt');
fprintf(file_,'%d\n',new_list);
fclose(file_);
disp('Done')
%% Verify on the subtrain: 
I=find(list20~=-1);
[~,J]=ismember(train_names,full_train_names);
JJ=ismember(J,I);
%
sub_old_labels=list20(J);
sub_new_labels=bls6(J);
%
% disp('5% confusion:');
% q11=sum((list5(J)==train_labels).*(train_labels==1))/sum(list5(J)==1)
% q00=sum((list5(J)==train_labels).*(train_labels==0))/sum(list5(J)==0)

% disp('20% confusion:')
% q11=sum((sub_old_labels==train_labels).*(train_labels==1))/sum(sub_old_labels==1)
% q00=sum((sub_old_labels==train_labels).*(train_labels==0))/sum(sub_old_labels==0)

disp('New confusion:')
q11=sum((sub_new_labels==train_labels).*(train_labels==1))/sum(sub_new_labels==1)
q00=sum((sub_new_labels==train_labels).*(train_labels==0))/sum(sub_new_labels==0)
%% Bootstrap performance:
%% Accuracies & losses:
close all
figure(1)
set(1,'units','normalized','position',[.1 .1 .5 .6])
set(1,'PaperType','A4','PaperPositionMode','auto')

subplot(3,1,1)
    plot(clean_log.data(:,1),clean_log.data(:,4),'col','k','linewidth',1);
    hold on,
    plot(b20.data(:,1),b20.data(:,4),'col',col(2,:),'linewidth',1.5);
    hold on,
    plot(s1.data(:,1),s1.data(:,4),'col',col(3,:),'linewidth',1.5);
    hold on,
    plot(bs2.data(:,1),bs2.data(:,4),'col',col(4,:),'linewidth',1.5);
    hold on,
    plot(bs3.data(:,1),bs3.data(:,4),'col',col(5,:),'linewidth',1.5);
    hold on,
    plot(bs4.data(:,1),bs4.data(:,4),'col',col(6,:),'linewidth',1.5);
    hold on,
    plot(bs5.data(:,1),bs5.data(:,4),'col',col(7,:),'linewidth',1.5);
    hold on,
    plot(bs6.data(:,1),bs6.data(:,4),'col',col(9,:),'linewidth',1.5);
%     hold on,
%     plot(bs6_crop.data(:,1),bs6_crop.data(:,4),'col',col(9,:),'linewidth',1.5);
    ylabel('Accuracy')
    ylim([.9 1])
    legend({'clean','b20','strap 1','strap 2','strap 3','strap 4','strap 5','strap 6'},'location','eastoutside')
    
%Losses
subplot(3,1,2)
    %Test loss:
    plot(clean_log.data(:,1),clean_log.data(:,5),'col','k','linewidth',1);
    hold on,
    plot(b20.data(:,1),b20.data(:,5),'col',col(2,:),'linewidth',1.5);
    hold on,
    plot(s1.data(:,1),s1.data(:,5),'col',col(3,:),'linewidth',1.5);
    hold on,
    plot(bs2.data(:,1),bs2.data(:,5),'col',col(4,:),'linewidth',1.5);
    hold on,
    plot(bs3.data(:,1),bs3.data(:,5),'col',col(5,:),'linewidth',1.5);
    hold on,
    plot(bs4.data(:,1),bs4.data(:,5),'col',col(6,:),'linewidth',1.5);
    hold on,
    plot(bs5.data(:,1),bs5.data(:,5),'col',col(7,:),'linewidth',1.5);
    hold on,
    plot(bs6.data(:,1),bs6.data(:,5),'col',col(9,:),'linewidth',1.5);
%     hold on,
%     plot(bs6_crop.data(:,1),bs6_crop.data(:,5),'col',col(9,:),'linewidth',1.5);
    ylabel('Test loss')
    %ylim([.8 1])
   % legend({'b20','strap 1','strap 2','strap 3','strap 4'},'location','northwest')
    
    
subplot(3,1,3)
    %Train loss:
    plot(b20T.data(:,1),b20T.data(:,4),'col',col(2,:),'linewidth',1.5);
    hold on,
    plot(s1T.data(:,1),s1T.data(:,4),'col',col(3,:),'linewidth',1.5);
    hold on,
    plot(bs2T.data(:,1),bs2T.data(:,4),'col',col(4,:),'linewidth',1.5);
    hold on,
    plot(bs3T.data(:,1),bs3T.data(:,4),'col',col(5,:),'linewidth',1.5);
    hold on,
    plot(bs4T.data(:,1),bs4T.data(:,4),'col',col(6,:),'linewidth',1.5);
    hold on,
    plot(bs5T.data(:,1),bs5T.data(:,4),'col',col(7,:),'linewidth',1.5);
    hold on,
    plot(bs6T.data(:,1),bs6T.data(:,4),'col',col(9,:),'linewidth',1.5);
%     hold on,
%     plot(bs6T_crop.data(:,1),bs6T_crop.data(:,4),'col',col(9,:),'linewidth',1.5);
    ylabel('Train loss')
    %legend({'b20','strap 1','strap 2','strap 3','strap 4'},'location','northwest')
    xlabel('Iteration')

print 'figures/balanced/tocrop/strap2_acc_loss' '-dpdf'
%% Roc 
close all; clc;
figure(1)
legends={};
% Train set:    
% baseline 20%
    I=find(list20~=-1);
    [scores,pred]=max(feat20T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*list20(I)-1,scores);
    legends{end+1}=sprintf('baseline 20 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(2,:),'linewidth',1.5)
    hold on,
    
% strap 1
    I=find(ls1~=-1);
    [scores,pred]=max(strap1T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*ls1(I)-1,scores);
    legends{end+1}=sprintf('strap1 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(3,:),'linewidth',1.5)
    hold on,
    
% strap 2
    I=find(bls2~=-1);
    [scores,pred]=max(bstrap2T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*bls2(I)-1,scores);
    legends{end+1}=sprintf('strap2 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(4,:),'linewidth',1.5)
    hold on,

% strap 3
    I=find(bls3~=-1);
    [scores,pred]=max(bstrap3T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*bls3(I)-1,scores);
    legends{end+1}=sprintf('strap3 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(5,:),'linewidth',1.5)
    hold on,   
    
% strap 4
    I=find(bls4~=-1);
    [scores,pred]=max(bstrap4T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*bls4(I)-1,scores);
    legends{end+1}=sprintf('strap4 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(6,:),'linewidth',1.5)
    hold on,   

% strap 5
    I=find(bls5~=-1);
    [scores,pred]=max(bstrap5T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*bls5(I)-1,scores);
    legends{end+1}=sprintf('strap5 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(7,:),'linewidth',1.5)
    hold on, 
    
% strap 6
    I=find(bls6~=-1);
    [scores,pred]=max(bstrap6T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*bls6(I)-1,scores);
    legends{end+1}=sprintf('strap6 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(9,:),'linewidth',1.5)
    hold on, 
axis square
xlabel('false positives rate')
ylabel('true positives rate')
legend(legends,'fontsize',12,'location','southeast');
%title('Train set','fontsize' ,14)
print 'figures/balanced/tocrop/strap2_roc_train' '-dpdf'

figure(2)
legends={};    
% Test set:
 % baseline 20%
    [scores,pred]=max(feat20,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('baseline 20 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(2,:),'linewidth',1.5)
    hold on,

% strap 1
    [scores,pred]=max(strap1,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('strap1 auc=%.3f',info.auc);
    %fnr=1-tpr;
    %fpr=1-tnr;
    plot(1-tnr, tpr, 'color',col(3,:),'linewidth',1.5)


% strap 2 - mean
    [scores,pred]=max(bstrap2,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('strap2 auc=%.3f',info.auc);
    plot(1-tnr, tpr,'color',col(4,:),'linewidth',1.5)
    
% strap 3 - mean
    [scores,pred]=max(bstrap3,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('strap3 auc=%.3f',info.auc);
    plot(1-tnr, tpr,'color',col(5,:),'linewidth',1.5)
    
% strap 4 - mean
    [scores,pred]=max(bstrap4,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('strap4 auc=%.3f',info.auc);
    plot(1-tnr, tpr,'color',col(6,:),'linewidth',1.5)
    
% strap 5 - mean
    [scores,pred]=max(bstrap5,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('strap5 auc=%.3f',info.auc);
    plot(1-tnr, tpr,'color',col(7,:),'linewidth',1.5)
    
% strap 6 - mean
    [scores,pred]=max(bstrap6,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('strap6 auc=%.3f',info.auc);
    plot(1-tnr, tpr,'color',col(9,:),'linewidth',1.5)   
    
    
% % strap 6 - mean-crop
%     [scores,pred]=max(bstrap6_crop,[],2);
%     pred=2*pred-3;
%     scores=scores.*pred;
%     [tpr,tnr,info]=vl_roc(labels,scores);
%     legends{end+1}=sprintf('strap6 -crop auc=%.3f',info.auc);
%     plot(1-tnr, tpr,'color',col(9,:),'linewidth',1.5)   
%     
% Clean
    [scores,pred]=max(clean,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('Test clean(20%%) auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color','k','linewidth',1)

    
    
axis square
xlabel('false positives rate')
ylabel('true positives rate')
legend(legends,'fontsize',12,'location','southeast');
%title('Test set','fontsize' ,14)
print 'figures/balanced/tocrop/strap2_roc_test' '-dpdf'
%% Rapport
acc2=[clean_log.data(end,4)...
    b5.data(end,4)...
    s1.data(end,4), bs2.data(end,4), bs3.data(end,4), bs4.data(end,4),...
    bs5.data(end,4), bs6.data(end,4)];
figure(2)
legends={};   
auc2=[];
ap2=[]; ar2=[];
i=1;
% Test set:
 % baseline 20%
     disp('baseline')
    [scores,pred]=max(feat20,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [~,~,info]=vl_roc(labels,scores);
    auc2(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col',col(i,:),'plotBaseline',0);
    %ap2(end+1)=mean(PREC);
    [~,~,info]=vl_pr(labels,scores);
    ap2(end+1)=info.ap;
    ar2(end+1)=mean(TPR);
    legends{end+1}=sprintf('iter 0');
    i=i+1;

% strap 1
     disp('strap 1')
    [scores,pred]=max(strap1,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [~,~,info]=vl_roc(labels,scores);
    auc2(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col',col(i,:),'holdFigure',1);
    %ap2(end+1)=mean(PREC);
    [~,~,info]=vl_pr(labels,scores);
    ap2(end+1)=info.ap;
    ar2(end+1)=mean(TPR);
    legends{end+1}=sprintf('iter 1');
    i=i+1;
% strap 2
    disp('strap 2')
    [scores,pred]=max(bstrap2,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [~,~,info]=vl_roc(labels,scores);
    auc2(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col',col(i,:),'holdFigure',1);
    %ap2(end+1)=mean(PREC);
    [~,~,info]=vl_pr(labels,scores);
    ap2(end+1)=info.ap;
    ar2(end+1)=mean(TPR);
    legends{end+1}=sprintf('iter 2');
    i=i+1;
% strap 3
    disp('strap 3')
    [scores,pred]=max(bstrap3,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [~,~,info]=vl_roc(labels,scores);
    auc2(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col',col(i,:),'holdFigure',1);
    %ap2(end+1)=mean(PREC);
    [~,~,info]=vl_pr(labels,scores);
    ap2(end+1)=info.ap;
    ar2(end+1)=mean(TPR);
    legends{end+1}=sprintf('iter 3');
    i=i+1;
% strap 4
    disp('strap 4')
    [scores,pred]=max(bstrap4,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [~,~,info]=vl_roc(labels,scores);
    auc2(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col',col(i,:),'holdFigure',1);
    %ap2(end+1)=mean(PREC);
    [~,~,info]=vl_pr(labels,scores);
    ap2(end+1)=info.ap;
    ar2(end+1)=mean(TPR);
    legends{end+1}=sprintf('iter 4');
    i=i+1;
 % strap 5
    disp('strap 5')
    [scores,pred]=max(bstrap5,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [~,~,info]=vl_roc(labels,scores);
    auc2(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col',col(i,:),'holdFigure',1);
    %ap2(end+1)=mean(PREC);
    [~,~,info]=vl_pr(labels,scores);
    ap2(end+1)=info.ap;
    ar2(end+1)=mean(TPR);
    legends{end+1}=sprintf('iter 5');
    i=i+1;
% strap 6
    disp('strap 6')
    [scores,pred]=max(bstrap6,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [~,~,info]=vl_roc(labels,scores);
    auc2(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col',col(i,:),'holdFigure',1);
    %ap2(end+1)=mean(PREC);
    [~,~,info]=vl_pr(labels,scores);
    ap2(end+1)=info.ap;
    ar2(end+1)=mean(TPR);
    legends{end+1}=sprintf('iter 6');
    i=i+1;
% Clean
    disp('Clean')
    [scores,pred]=max(clean,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [~,~,info]=vl_roc(labels,scores);
    auc2(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col','k','holdFigure',1);
    %ap2(end+1)=mean(PREC);
    [~,~,info]=vl_pr(labels,scores);
    ap2(end+1)=info.ap;
    ar2(end+1)=mean(TPR);
    legends{end+1}=sprintf('Clean');

axis square
% xlim([0 .5])
% ylim([.5 1])
xlabel('false positives rate')
ylabel('true positives rate')
legend(legends,'fontsize',12,'location','southwest');
%title('Test set','fontsize' ,14)
print 'figures/rapport/tocrop/pr' '-dpdf'

%%
sz2=0:6;
sz=0:7;
figure(3), clf
set(3,'units','normalized','position',[.1 .1 .6 .7])
set(3,'PaperType','A4','PaperPositionMode','auto','PaperOrientation','landscape')

subplot(3,1,1)
plot(sz,ap(1:end-1),'-ob','MarkerSize',5,'MarkerFaceColor','b');
hold on
plot(sz2,ap2(1:end-1),'-or','MarkerSize',5,'MarkerFaceColor','r');
line([0 7],[ap2(end) ap2(end)],'linestyle','--','color','k');
title('Average precision')
ylabel('Average precision')
grid on,
grid minor,

subplot(3,1,2)
plot(sz,auc(1:end-1),'-ob','MarkerSize',5,'MarkerFaceColor','b');
hold on
plot(sz2,auc2(1:end-1),'-or','MarkerSize',5,'MarkerFaceColor','r');
line([0 7],[auc2(end) auc2(end)],'linestyle','--','color','k');
title('AUC')
ylabel('AUC')
grid on,
grid minor,

subplot(3,1,3)
plot(sz,acc(2:end),'-ob','MarkerSize',5,'MarkerFaceColor','b');
hold on
plot(sz2,acc2(2:end),'-or','MarkerSize',5,'MarkerFaceColor','r');
line([0 7],[acc(1) acc(1)],'linestyle','--','color','k');
title('Accuracy')
xlabel('Bootstrap iteration')
ylabel('Accuracy')
grid on,
grid minor,
legend({'Bootstrap 1','Bootstrap 2','Clean'},'location','southwest','orientation','horizontal')
print 'figures/rapport/tocrop/ap_auc_2' '-dpdf'