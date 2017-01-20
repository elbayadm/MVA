%% Settings
clear; clc; close all;
warning 'off'
addpath(genpath('Matlab/'))
addpath(genpath('logs/'))
addpath(genpath('features/'))
addpath('Confusion/')
addpath('data/morpho_QQ/original')
addpath('/Users/redgns/GitHub/Morpho/OldClassifier')
set(0, 'DefaultAxesFontName', 'Helvetica')
set(0,'defaultTextFontName', 'Helvetica')
set(0,'DefaultAxesFontSize',14)
set(0,'DefaultTextFontSize',14)
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
% f1=importdata('morpho/NLTH30_30/fix/nodp.log.test');
% f2=importdata('morpho/NLTH30_30/fix/fix2.log.test');
% f3=importdata('morpho/NLTH30_30/fix/fix3.log.test');
% f1T=importdata('morpho/NLTH30_30/fix/nodp.log.train');
% f2T=importdata('morpho/NLTH30_30/fix/fix2.log.train');
% f3T=importdata('morpho/NLTH30_30/fix/fix3.log.train');

s1=importdata('morpho/NLTH30_30/bs1/strap1.log.test');
s1T=importdata('morpho/NLTH30_30/bs1/strap1.log.train');
s2=importdata('morpho/NLTH30_30/bs1/strap2.log.test');
s2T=importdata('morpho/NLTH30_30/bs1/strap2.log.train');
s3=importdata('morpho/NLTH30_30/bs1/strap3.log.test');
s3T=importdata('morpho/NLTH30_30/bs1/strap3.log.train');
s4=importdata('morpho/NLTH30_30/bs1/strap4.log.test');
s4T=importdata('morpho/NLTH30_30/bs1/strap4.log.train');
s5=importdata('morpho/NLTH30_30/bs1/strap5.log.test');
s5T=importdata('morpho/NLTH30_30/bs1/strap5.log.train');
s6=importdata('morpho/NLTH30_30/bs1/strap6.log.test');
s6T=importdata('morpho/NLTH30_30/bs1/strap6.log.train');
s7=importdata('morpho/NLTH30_30/bs1/strap7.log.test');
s7T=importdata('morpho/NLTH30_30/bs1/strap7.log.train'); 

% Features:
% train:
feat20T=importdata('NLTH30_30/train_bprob_10000.txt');
feat5T=importdata('NLTH30_30/train_prob_10000.txt');
% fix1T=importdata('NLTH30_30/fix/train_nodp_10000.txt');
% fix2T=importdata('NLTH30_30/fix/train_fix2_10000.txt');
% fix3T=importdata('NLTH30_30/fix/train_fix3_10000.txt');

strap1T=importdata('NLTH30_30/bs1/train_strap1_10000.txt');
strap2T=importdata('NLTH30_30/bs1/train_strap2_10000.txt');
strap3T=importdata('NLTH30_30/bs1/train_strap3_10000.txt');
strap4T=importdata('NLTH30_30/bs1/train_strap4_10000.txt');
strap5T=importdata('NLTH30_30/bs1/train_strap5_10000.txt');
strap6T=importdata('NLTH30_30/bs1/train_strap6_10000.txt');
strap7T=importdata('NLTH30_30/bs1/train_strap7_10000.txt');

% test:

% fix1=importdata('NLTH30_30/fix/test_nodp_10000.txt');
% fix2=importdata('NLTH30_30/fix/test_fix2_10000.txt');
% fix3=importdata('NLTH30_30/fix/test_fix3_10000.txt');

feat20=importdata('NLTH30_30/bprob_10000.txt');
feat5=importdata('NLTH30_30/prob_10000.txt');
clean=importdata('morpho/NL000/prob1_10000.txt');

strap1=importdata('NLTH30_30/bs1/test_strap1_10000.txt');
strap2=importdata('NLTH30_30/bs1/test_strap2_10000.txt');
strap3=importdata('NLTH30_30/bs1/test_strap3_10000.txt');
strap4=importdata('NLTH30_30/bs1/test_strap4_10000.txt');
strap5=importdata('NLTH30_30/bs1/test_strap5_10000.txt');
strap6=importdata('NLTH30_30/bs1/test_strap6_10000.txt');
strap7=importdata('NLTH30_30/bs1/test_strap7_10000.txt');

% labels
list20=importdata('balanced_labels_30_30.txt');
list5=importdata('labels_30_30.txt');
labelsTT=importdata('out_glasses_lfw.txt');
labelsTT=2*labelsTT.data-1;
ls1=importdata('strap1_balanced_labels_30_30.txt');
ls2=importdata('strap2_balanced_labels_30_30.txt');
ls3=importdata('strap3_balanced_labels_30_30.txt');
ls4=importdata('strap4_balanced_labels_30_30.txt');
ls5=importdata('strap5_balanced_labels_30_30.txt');
ls6=importdata('strap6_balanced_labels_30_30.txt');
ls7=importdata('strap7_balanced_labels_30_30.txt');

% Bootstrapping scores:
s1scores=importdata('strap1_balanced_scores_30_30.txt');
s2scores=importdata('strap2_balanced_scores_30_30.txt');
s3scores=importdata('strap3_balanced_scores_30_30.txt');
s4scores=importdata('strap4_balanced_scores_30_30.txt');
s5scores=importdata('strap5_balanced_scores_30_30.txt');
s6scores=importdata('strap6_balanced_scores_30_30.txt');
s7scores=importdata('strap7_balanced_scores_30_30.txt');

% Old scores:
full_train=importdata('full_train.txt');
full_train_scores=full_train.data;
full_train_names=full_train.textdata;

%Train subset:
train=importdata('subtrain_os.txt');
train_names=train.textdata;
train_labels=train.data(:,1);
train_scores=train.data(:,2);
clear train
%% Fix the baseline 20% 
%% Accuracies & Losses
close all; clc;
NL={'TH30_30'};
nr=length(NL);
figure(1)
set(1,'units','normalized','position',[.1 .1 .5 .6*nr])
set(1,'PaperType','A4','PaperPositionMode','auto')
for i=1:nr
    %Accuracies
    subplot(3*nr,1,i)
    plot(b5.data(:,1),b5.data(:,4),'col',col(1,:),'linewidth',1.5);
    hold on,
    plot(b20.data(:,1),b20.data(:,4),'col',col(2,:),'linewidth',1.5);
    hold on,
    plot(f1.data(:,1),f1.data(:,4),'col',col(3,:),'linewidth',1.5);
    hold on,
    plot(f2.data(:,1),f2.data(:,4),'col',col(4,:),'linewidth',1.5);
    hold on,
    plot(f3.data(:,1),f3.data(:,4),'col',col(5,:),'linewidth',1.5);
    
    ylabel('Accuracy')
    ylim([.8 1])
    legend({'b5','b20','b5 w/o dropout','b20 w/o dropout','b20 Adagrad'},'location','southwest')
    
    %Losses
    
    subplot(3*nr,1,2*i)
    %Test loss:
    plot(b5.data(:,1),b5.data(:,5),'col',col(1,:),'linewidth',1.5);
    hold on,
    plot(b20.data(:,1),b20.data(:,5),'col',col(2,:),'linewidth',1.5);
    hold on,
    plot(f1.data(:,1),f1.data(:,5),'col',col(3,:),'linewidth',1.5);
    hold on,
    plot(f2.data(:,1),f2.data(:,5),'col',col(4,:),'linewidth',1.5);
    hold on,
    plot(f3.data(:,1),f3.data(:,5),'col',col(5,:),'linewidth',1.5);
    
    ylabel('Test loss')
    %ylim([.8 1])
    legend({'b5','b20','b5 w/o dropout','b20 w/o dropout','b20 Adagrad'},'location','northwest')
    
    
    subplot(3*nr,1,3*i)
    %Train loss:
    plot(b5T.data(:,1),b5T.data(:,4),'col',col(1,:),'linewidth',1.5);
    hold on,
    plot(b20T.data(:,1),b20T.data(:,4),'col',col(2,:),'linewidth',1.5);
    hold on,
    plot(f1T.data(:,1),f1T.data(:,4),'col',col(3,:),'linewidth',1.5);
    hold on,
    plot(f2T.data(:,1),f2T.data(:,4),'col',col(4,:),'linewidth',1.5);
    hold on,
    plot(f3T.data(:,1),f3T.data(:,4),'col',col(5,:),'linewidth',1.5);
    ylabel('Loss')
    %ylim([.8 1])
    legend({'b5','b20','b5 w/o dropout','b20 w/o dropout','b20 Adagrad'},'location','northwest')
end
xlabel('Iteration')
print 'figures/balanced/tocrop/acc_loss' '-dpdf'
%% ROC curves
close all; clc;
figure,
legends={};
% Train set:

% baseline 5%
    I=find(list5~=-1);
    [scores,pred]=max(feat5T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*list5(I)-1,scores);
    legends{end+1}=sprintf('Train b5 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(1,:),'linewidth',1.5)
    hold on,

% baseline 20%
    I=find(list20~=-1);
    [scores,pred]=max(feat20T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*list20(I)-1,scores);
    legends{end+1}=sprintf(' Train b20 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(2,:),'linewidth',1.5)
    hold on,

% baseline 5% no dp
    I=find(list5~=-1);
    [scores,pred]=max(fix1T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*list5(I)-1,scores);
    legends{end+1}=sprintf('Train b5 w/o dropout auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(3,:),'linewidth',1.5)
    hold on,

% fix2
    [scores,pred]=max(fix2T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*list20(I)-1,scores);
    legends{end+1}=sprintf('Train b20 w/o dropout auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(4,:),'linewidth',1.5)
    hold on,
%fix3
    [scores,pred]=max(fix3T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*list20(I)-1,scores);
    legends{end+1}=sprintf('Train b20 Adagrad auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(5,:),'linewidth',1.5)
    hold on,


% Test set:
% baseline 5%
    [scores,pred]=max(feat5,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('Test b5 auc=%.3f',info.auc);
    plot(1-tnr, tpr,'--', 'color',col(1,:),'linewidth',1.5)

% baseline 20%
    [scores,pred]=max(feat20,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('Test b20 auc=%.3f',info.auc);
    plot(1-tnr, tpr,'--', 'color',col(2,:),'linewidth',1.5)
    hold on,

% baseline 5% nodp
    [scores,pred]=max(fix1,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('Test b5 w/o dropout auc=%.3f',info.auc);
    plot(1-tnr, tpr,'--', 'color',col(3,:),'linewidth',1.5)
    hold on,
    
% fix2
    [scores,pred]=max(fix2,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('Test b20 w/o dropout auc=%.3f',info.auc);
    plot(1-tnr, tpr, '--','color',col(4,:),'linewidth',1.5)
    hold on,
% fix3
    [scores,pred]=max(fix3,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('Test b20 Adagrad auc=%.3f',info.auc);
    plot(1-tnr, tpr, '--','color',col(5,:),'linewidth',1.5)

legend(legends,'fontsize',12,'location','southeast');
print 'figures/balanced/tocrop/roc' '-dpdf'
%% hist of scores - Test:
    close all,
    figure,
    
    subplot(4,1,1)
    [scores,pred]=max(feat5,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    h1=histogram(scores,100);
    set(h1,'FaceColor',col(1,:),'facealpha',.75);
    title('baseline 5')
    
    subplot(4,1,2)
    [scores,pred]=max(feat20,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    h0=histogram(scores,100);
    set(h0,'FaceColor',col(2,:),'facealpha',.75);
    title('baseline 20')

    subplot(4,1,3)
    [scores,pred]=max(fix2,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    h1=histogram(scores,100);
    set(h1,'FaceColor',col(8,:),'facealpha',.75);
    title('baseline 20 w/o dropout')

    subplot(4,1,4)
    [scores,pred]=max(fix3,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    h1=histogram(scores,100);
    set(h1,'FaceColor',col(10,:),'facealpha',.75);
    title('baseline 20 AdaGrad')
    print 'figures/balanced/tocrop/test_scores' '-dpdf'
%% hist of scores - Train:
    close all,
    figure,
    
    subplot(4,1,1)
    [scores,pred]=max(feat5T,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    h1=histogram(scores,100);
    set(h1,'FaceColor',col(1,:),'facealpha',.75);
    title('baseline 5')
    
    subplot(4,1,2)
    [scores,pred]=max(feat20T,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    h0=histogram(scores,100);
    set(h0,'FaceColor',col(2,:),'facealpha',.75);
    title('baseline 20')

    subplot(4,1,3)
    [scores,pred]=max(fix2T,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    h1=histogram(scores,100);
    set(h1,'FaceColor',col(8,:),'facealpha',.75);
    title('baseline 20 w/o dropout')

    subplot(4,1,4)
    [scores,pred]=max(fix3T,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    h1=histogram(scores,100);
    set(h1,'FaceColor',col(10,:),'facealpha',.75);
    title('baseline 20 AdaGrad')
    print 'figures/balanced/tocrop/train_scores' '-dpdf'
%% Misclassified on the training 
I=find(list20~=-1);
names=full_train_names(I);
old_scores=[full_train_scores(I), s1scores(I), s2scores(I), s3scores(I),...
    s4scores(I),s5scores(I),s6scores(I),s7scores(I)];

% Training labels
true_labels=2*ls7(I)-1;

% Model scores
[scores,pred]=max(strap7T(I,:),[],2);
pred=2*pred-3;
scores=scores.*pred;

FP=find((true_labels==-1).*(pred==1));
FN=find((true_labels==1).*(pred==-1));
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
    text(50,70,sprintf('%.2f//%d/%d/%d/\n%d/%d/%d/%d/%d', scores(FP(i)),round(old_scores(FP(i),:))),...
       'background','w',...
       'verticalalignment','top', ...
       'horizontalalignment','center', ...
       'Margin',.1,...
       'fontsize', 5) ;
    set(gca,'xtick',[],'ytick',[]) ; axis image ;
  end
print 'figures/balanced/FNFP/FP_s7' '-dpdf'
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
    text(50,70,sprintf('%.2f//%d/%d/%d/\n%d/%d/%d/%d/%d', scores(FN(i)),round(old_scores(FN(i),:))),...
       'background','w',...
       'verticalalignment','top', ...
       'horizontalalignment','center', ...
       'Margin',.1,...
       'fontsize', 5) ;
    set(gca,'xtick',[],'ytick',[]) ; axis image ;
  end
  print 'figures/balanced/FNFP/FN_s7' '-dpdf'  
%% Bootstrapping 20%: - Build the new training labels
I=find(list20~=-1);
old_scores=s6scores(I);
cnn_scores=100*strap6T(I,2);
new_scores=(cnn_scores+old_scores)/2;
% figure,
% subplot(3,1,1)
% hist(old_scores,100);
% subplot(3,1,2)
% hist(cnn_scores,100);
% subplot(3,1,3)
% hist(new_scores,100);
th=50;  %(30+50)/2 %(40+50)/2 (45+50)/2~50
new_labels=new_scores>th;
new_list=list20;
new_list(I)=new_labels;
sum(new_labels==1)/length(new_labels)
sum(new_labels==0)/length(new_labels)

% Write them down:
% Scores
full_new_scores=-ones(size(full_train_scores));
full_new_scores(I)=new_scores;
file_ = fopen('data/morpho_QQ/strap7_balanced_scores_30_30.txt','wt');
fprintf(file_,'%.3f\n',full_new_scores);
fclose(file_);
disp('Done')
%% Labels
file_ = fopen('data/morpho_QQ/strap7_balanced_labels_30_30.txt','wt');
fprintf(file_,'%d\n',new_list);
fclose(file_);
disp('Done')
  %% Bootstrapping 05%: - Build the new training labels
I=find(list5~=-1);
old_scores=full_train_scores(I);
cnn_scores=100*feat5(I,2);
new_scores=(cnn_scores+old_scores)/2;
% figure,
% subplot(3,1,1)
% hist(old_scores,100);
% subplot(3,1,2)
% hist(cnn_scores,100);
% subplot(3,1,3)
% hist(new_scores,100);
th=45;  %(30+50)/2 %(40+50)/2
new_labels=new_scores>th;
new_list=list5;
new_list(I)=new_labels;
sum(new_labels==1)/length(new_labels)
sum(new_labels==0)/length(new_labels)
%% Write them down:
% Scores
full_new_scores=-ones(size(full_train_scores));
full_new_scores(I)=new_scores;
file_ = fopen('data/morpho_QQ/strap1_scores_30_30.txt','wt');
fprintf(file_,'%d\n',full_new_scores);
fclose(file_);
disp('Done')

% Labels
file_ = fopen('data/morpho_QQ/strap1_labels_30_30.txt','wt');
fprintf(file_,'%d\n',new_list);
fclose(file_);
disp('Done')
%% Verify on the subtrain: 
I=find(list20~=-1);
[~,J]=ismember(train_names,full_train_names);
JJ=ismember(J,I);
%
sub_old_labels=list20(J);
sub_new_labels=ls7(J);
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
    plot(b5.data(:,1),b5.data(:,4),'col',col(1,:),'linewidth',1.5);
    hold on,
    plot(b20.data(:,1),b20.data(:,4),'col',col(2,:),'linewidth',1.5);
    hold on,
    plot(s1.data(:,1),s1.data(:,4),'col',col(3,:),'linewidth',1.5);
    hold on,
    plot(s2.data(:,1),s2.data(:,4),'col',col(4,:),'linewidth',1.5);
    hold on,
    plot(s3.data(:,1),s3.data(:,4),'col',col(5,:),'linewidth',1.5);
    hold on,
    plot(s4.data(:,1),s4.data(:,4),'col',col(6,:),'linewidth',1.5);
    hold on,
    plot(s5.data(:,1),s5.data(:,4),'col',col(7,:),'linewidth',1.5);
    hold on,
    plot(s6.data(:,1),s6.data(:,4),'col',col(8,:),'linewidth',1.5);
    hold on,
    plot(s7.data(:,1),s7.data(:,4),'col',col(9,:),'linewidth',1.5);
    ylabel('Accuracy')
    ylim([.9 1])
    legend({'clean 20%','b5','b20','strap 1','strap 2','strap 3','strap 4',...
        'strap 5','strap 6', 'strap 7'},'location','eastoutside')
    
%Losses
subplot(3,1,2)
    %Test loss:
    plot(clean_log.data(:,1),clean_log.data(:,5),'col','k','linewidth',1);
    hold on,
    plot(b5.data(:,1),b5.data(:,5),'col',col(1,:),'linewidth',1.5);
    hold on,
    plot(b20.data(:,1),b20.data(:,5),'col',col(2,:),'linewidth',1.5);
    hold on,
    plot(s1.data(:,1),s1.data(:,5),'col',col(3,:),'linewidth',1.5);
    hold on,
    plot(s2.data(:,1),s2.data(:,5),'col',col(4,:),'linewidth',1.5);
    hold on,
    plot(s3.data(:,1),s3.data(:,5),'col',col(5,:),'linewidth',1.5);
    hold on,
    plot(s4.data(:,1),s4.data(:,5),'col',col(6,:),'linewidth',1.5);
    hold on,
    plot(s5.data(:,1),s5.data(:,5),'col',col(7,:),'linewidth',1.5);
    hold on,
    plot(s6.data(:,1),s6.data(:,5),'col',col(8,:),'linewidth',1.5);
    hold on,
    plot(s7.data(:,1),s7.data(:,5),'col',col(9,:),'linewidth',1.5);
    ylabel('Test loss')
    %ylim([.8 1])
%     legend({'b5','b20','strap 1','strap 2','strap 3','strap 4',...
%         'strap 5','strap 6', 'strap 7'},'location','northwest')
    
    
subplot(3,1,3)
    %Train loss:
    plot(b5T.data(:,1),b5T.data(:,4),'col',col(1,:),'linewidth',1.5);
    hold on,
    plot(b20T.data(:,1),b20T.data(:,4),'col',col(2,:),'linewidth',1.5);
    hold on,
    plot(s1T.data(:,1),s1T.data(:,4),'col',col(3,:),'linewidth',1.5);
    hold on,
    plot(s2T.data(:,1),s2T.data(:,4),'col',col(4,:),'linewidth',1.5);
    hold on,
    plot(s3T.data(:,1),s3T.data(:,4),'col',col(5,:),'linewidth',1.5);
    hold on,
    plot(s4T.data(:,1),s4T.data(:,4),'col',col(6,:),'linewidth',1.5);
    hold on,
    plot(s5T.data(:,1),s5T.data(:,4),'col',col(7,:),'linewidth',1.5);
    hold on,
    plot(s6T.data(:,1),s6T.data(:,4),'col',col(8,:),'linewidth',1.5);
    hold on,
    plot(s7T.data(:,1),s7T.data(:,4),'col',col(9,:),'linewidth',1.5);
    ylabel('Train loss')
%     legend({'b5','b20','strap 1','strap 2','strap 3','strap 4',...
%         'strap 5','strap 6', 'strap 7'},'location','northwest')
    xlabel('Iteration')

%print 'figures/balanced/tocrop/strap_acc_loss' '-dpdf'
%% Roc 
close all; clc;
figure(1)
legends={};
% Train set:
% baseline 5%
    I=find(list5~=-1);
    [scores,pred]=max(feat5T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*list5(I)-1,scores);
    legends{end+1}=sprintf('b5 auc=%.3f',info.auc);
    plot(1-tnr, tpr,'color',col(1,:),'linewidth',1.5)
    hold on,
    
% baseline 20%
    I=find(list20~=-1);
    [scores,pred]=max(feat20T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*list20(I)-1,scores);
    legends{end+1}=sprintf('b20 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(7,:),'linewidth',1.5)
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
    I=find(ls2~=-1);
    [scores,pred]=max(strap2T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*ls2(I)-1,scores);
    legends{end+1}=sprintf('strap2 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(4,:),'linewidth',1.5)
    hold on,

% strap 3
    I=find(ls3~=-1);
    [scores,pred]=max(strap3T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*ls3(I)-1,scores);
    legends{end+1}=sprintf('strap3 auc=%.3f',info.auc);
    plot(1-tnr, tpr,'color',col(5,:),'linewidth',1.5)
    hold on,

% strap 4
    I=find(ls4~=-1);
    [scores,pred]=max(strap4T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*ls4(I)-1,scores);
    legends{end+1}=sprintf('strap4 auc=%.3f',info.auc);
    plot(1-tnr, tpr,'color',col(6,:),'linewidth',1.5)
    hold on,
    
% strap 5
    I=find(ls5~=-1);
    [scores,pred]=max(strap5T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*ls5(I)-1,scores);
    legends{end+1}=sprintf('strap5 auc=%.3f',info.auc);
    plot(1-tnr, tpr,'color',col(2,:),'linewidth',1.5)
    hold on,
    
% strap 6
    I=find(ls6~=-1);
    [scores,pred]=max(strap6T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*ls6(I)-1,scores);
    legends{end+1}=sprintf('strap6 auc=%.3f',info.auc);
    plot(1-tnr, tpr,'color',col(8,:),'linewidth',1.5)   
    
% strap 7
    I=find(ls7~=-1);
    [scores,pred]=max(strap7T(I,:),[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(2*ls7(I)-1,scores);
    legends{end+1}=sprintf('strap7 auc=%.3f',info.auc);
    plot(1-tnr, tpr,'color',col(9,:),'linewidth',1.5)    

axis square
xlabel('false positives rate')
ylabel('true positives rate')
legend(legends,'fontsize',12,'location','southeast');
%title('Train set','fontsize' ,14)
print 'figures/balanced/tocrop/strap_roc_train' '-dpdf'

figure(2)
legends={};    
% Test set:
% baseline 5%
    [scores,pred]=max(feat5,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('b5 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(1,:),'linewidth',1.5)
    hold on,

 % baseline 20%
    [scores,pred]=max(feat20,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('b20 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(7,:),'linewidth',1.5)
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

% strap 2
    [scores,pred]=max(strap2,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('strap2 auc=%.3f',info.auc);
    plot(1-tnr, tpr,'color',col(4,:),'linewidth',1.5)
    
% strap 3
    [scores,pred]=max(strap3,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('strap3 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(5,:),'linewidth',1.5)

% strap 4
    [scores,pred]=max(strap4,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('strap4 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(6,:),'linewidth',1.5)
    
 % strap 5
    [scores,pred]=max(strap5,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('strap5 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(2,:),'linewidth',1.5)

% strap 6
    [scores,pred]=max(strap6,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('strap6 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(8,:),'linewidth',1.5)

% strap 7
    [scores,pred]=max(strap7,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    legends{end+1}=sprintf('strap7 auc=%.3f',info.auc);
    plot(1-tnr, tpr, 'color',col(9,:),'linewidth',1.5)
    
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
print 'figures/balanced/tocrop/strap_roc_test' '-dpdf'

%% Rapport v2 - accuracy
close all
figure(1)
set(1,'units','normalized','position',[.1 .1 .5 .2])
set(1,'PaperType','A4','PaperPositionMode','auto')
acc=[clean_log.data(end,4)...
    b5.data(end,4)...
    s1.data(end,4), s2.data(end,4), s3.data(end,4), s4.data(end,4),...
    s5.data(end,4), s6.data(end,4), s7.data(end,4)];

plot(0:7,acc(2:end),'ob','MarkerSize',5,'MarkerFaceColor','b');
line([0 7],[acc(1) acc(1)],'linestyle','--','color','k');
ylabel('accuracy')
xlabel('bootstrap iteration')
%%

figure(2)
legends={};   
auc=[];
ap=[]; ar=[];
i=1;
% Test set:
 % baseline 20%
     disp('baseline')
    [scores,pred]=max(feat20,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [~,~,info]=vl_roc(labels,scores);
    auc(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',1,'plotPR',0,'col',col(i,:),'plotBaseline',0);
    [~,~,info]=vl_pr(labels,scores);
    ap(end+1)=info.ap;
    ar(end+1)=mean(TPR);
    legends{end+1}=sprintf('iter 0');
    i=i+1;

% strap 1
     disp('strap 1')
    [scores,pred]=max(strap1,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [~,~,info]=vl_roc(labels,scores);
    auc(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',1,'plotPR',0,'col',col(i,:),'holdFigure',1);
    [~,~,info]=vl_pr(labels,scores);
    ap(end+1)=info.ap;
    ar(end+1)=mean(TPR);
    legends{end+1}=sprintf('iter 1');
    i=i+1;
% strap 2
    disp('strap 2')
    [scores,pred]=max(strap2,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [~,~,info]=vl_roc(labels,scores);
    auc(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',1,'plotPR',0,'col',col(i,:),'holdFigure',1);
    [~,~,info]=vl_pr(labels,scores);
    ap(end+1)=info.ap;
    ar(end+1)=mean(TPR);
    legends{end+1}=sprintf('iter 2');
    i=i+1;
% strap 3
    disp('strap 3')
    [scores,pred]=max(strap3,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [~,~,info]=vl_roc(labels,scores);
    auc(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',1,'plotPR',0,'col',col(i,:),'holdFigure',1);
    [~,~,info]=vl_pr(labels,scores);
    ap(end+1)=info.ap;
    ar(end+1)=mean(TPR);
    legends{end+1}=sprintf('iter 3');
    i=i+1;
% strap 4
    disp('strap 4')
    [scores,pred]=max(strap4,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [~,~,info]=vl_roc(labels,scores);
    auc(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',1,'plotPR',0,'col',col(i,:),'holdFigure',1);
    [~,~,info]=vl_pr(labels,scores);
    ap(end+1)=info.ap;
    ar(end+1)=mean(TPR);
    legends{end+1}=sprintf('iter 4');
    i=i+1;
 % strap 5
    disp('strap 5')
    [scores,pred]=max(strap5,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [~,~,info]=vl_roc(labels,scores);
    auc(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',1,'plotPR',0,'col',col(i,:),'holdFigure',1);
    [~,~,info]=vl_pr(labels,scores);
    ap(end+1)=info.ap;
    ar(end+1)=mean(TPR);
    legends{end+1}=sprintf('iter 5');
    i=i+1;
% strap 6
    disp('strap 6')
    [scores,pred]=max(strap6,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [~,~,info]=vl_roc(labels,scores);
    auc(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',1,'plotPR',0,'col',col(i,:),'holdFigure',1);
    [~,~,info]=vl_pr(labels,scores);
    ap(end+1)=info.ap;
    ar(end+1)=mean(TPR);
    legends{end+1}=sprintf('iter 6');
    i=i+1;
% strap 7
    disp('strap 7')
    [scores,pred]=max(strap7,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [~,~,info]=vl_roc(labels,scores);
    auc(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',1,'plotPR',0,'col',col(i,:),'holdFigure',1);
    [~,~,info]=vl_pr(labels,scores);
    ap(end+1)=info.ap;
    ar(end+1)=mean(TPR);
    legends{end+1}=sprintf('iter 7');
    i=i+1;
% Clean
    disp('Clean')
    [scores,pred]=max(clean,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [~,~,info]=vl_roc(labels,scores);
    auc(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',1,'plotPR',0,'col','k','holdFigure',1);
    [~,~,info]=vl_pr(labels,scores);
    ap(end+1)=info.ap;
    ar(end+1)=mean(TPR);
    legends{end+1}=sprintf('Clean');

axis square
xlim([0 .5])
ylim([.5 1])
xlabel('false positives rate')
ylabel('true positives rate')
legend(legends,'fontsize',12,'location','southwest');
%title('Test set','fontsize' ,14)
%print 'figures/rapport/tocrop/roc' '-dpdf'

%%
sz=0:7;
figure(3), clf
set(3,'units','normalized','position',[.1 .1 .6 .7])
set(3,'PaperType','A4','PaperPositionMode','auto','PaperOrientation','landscape')

subplot(3,1,1)
plot(sz,ap(1:end-1),'-ob','MarkerSize',5,'MarkerFaceColor','b');
line([0 7],[ap(end) ap(end)],'linestyle','--','color','k');
title('Average precision')
ylabel('Average precision')
grid on,
grid minor,

subplot(3,1,2)
plot(sz,auc(1:end-1),'-ob','MarkerSize',5,'MarkerFaceColor','b');
line([0 7],[auc(end) auc(end)],'linestyle','--','color','k');
title('AUC')
ylabel('AUC')
grid on,
grid minor,

subplot(3,1,3)
plot(sz,acc(2:end),'-ob','MarkerSize',5,'MarkerFaceColor','b');
line([0 7],[acc(1) acc(1)],'linestyle','--','color','k');
title('Accuracy')
xlabel('Bootstrap iteration')
ylabel('Accuracy')
grid on,
grid minor,
legend({'Bootstrap','Clean'},'location','southeast')
print 'figures/rapport/tocrop/ap_auc' '-dpdf'