%% Settings
clear; close all; clc;
warning 'off'

addpath(genpath('Matlab/'))
addpath(genpath('logs/'))
addpath(genpath('features/'))
addpath('Confusion/')
addpath('data/morpho_QQ')

set(0, 'DefaultAxesFontName', 'Helvetica')
set(0,'defaultTextFontName', 'Helvetica')
set(0,'DefaultAxesFontSize',12)
set(0,'DefaultTextFontSize',12)

rand('seed',178);
col=[.7*hsv(12);.5*autumn(4)];
col=col(randperm(size(col,1),size(col,1)),:);
labels=importdata('test_labels.txt');
labels=2*labels.data-1;
% figure,
% for i=1:size(col,1)
%     plot(1:10,i*ones(1,10),'color',col(i,:))
%     hold on
% end
%% 20%
%% Baseline
%%
close all; clc;
NL={'TH10_30','TH20_20','TH30_30','TH40_40'};
nr=length(NL);
% Accuracies:
figure(1)
set(1,'units','normalized','position',[.1 .1 .55 .35])
set(1,'PaperType','A4','PaperPositionMode','auto')
for i=1:nr
    subplot(2,2,i)
    p1=importdata(sprintf('morpho/NL%s/p1.log.test',NL{i}));
    p2=importdata(sprintf('morpho/NL%s/bp1.log.test',NL{i}));
    plot(p1.data(:,1),p1.data(:,4),'col',col(1,:),'linewidth',2);
    hold on,
    plot(p2.data(:,1),p2.data(:,4),'col',col(6,:),'linewidth',2);
    ylabel('Accuracy')
    ylim([.8 1])
    legend({'baseline 5%','baseline 20%'},'location','southwest','fontsize',10)
    title(sprintf('N%d',i))
    if i<3
        set(gca,'XTickLabel',{});
    end
end
xlabel('Iteration')
print 'figures/tocrop/acc_balanced' '-dpdf'
%% ROC Curve:
close all; clc
auc=zeros(1,nr+1);
figure,
legends={};
for i=1:nr
    bprob=importdata(sprintf('morpho/NL%s/bprob_10000.txt',NL{i}));
    prob=importdata(sprintf('morpho/NL%s/prob_10000.txt',NL{i}));
    if i==1
        Cprob=importdata('morpho/NL000/prob1_10000.txt');
        % Clean
        [scores,pred]=max(Cprob,[],2);
        pred=2*pred-3;
        scores=scores.*pred;
        [tpr,tnr,info]=vl_roc(labels,scores);
        info.auc
        plot(1-tnr, tpr, 'color','k','linewidth',1.5)
        legends{end+1}='clean';
        hold on
        % all 0
        [tpr,tnr,info]=vl_roc(labels,-1*ones(1,length(labels)));
        auc(4)=info.auc;
        plot(1-tnr, tpr, 'color','r','linewidth',1.5)
        legends{end+1}='All 0';
    end
    
    % Baseline 5%
    [scores,pred]=max(prob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    info.auc
    plot(1-tnr, tpr,'--', 'color',col(i+mod(i,2)*2+1,:),'linewidth',1.5)
    legends{end+1}=sprintf('baseline 5%% N%d',i);
    hold on,

    % Balanced (20%)
    [scores,pred]=max(bprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    info.auc
    plot(1-tnr, tpr, 'color',col(i+mod(i,2)*2+1,:),'linewidth',1.5)
    hold on,
    legends{end+1}=sprintf('baseline 20%% N%d',i);
end
legend(legends,'location','southwest','fontsize',12,'location','southeast')
print 'figures/tocrop/roc_baseline_balanced' '-dpdf'

%% ROC Curve - Infogain
close all; clc
figure,
legends={};
for i=1:nr
    bprob=importdata(sprintf('morpho/NL%s/bprob_10000.txt',NL{i}));
    Hbprob=importdata(sprintf('morpho/NL%s/H2bprob_10000.txt',NL{i}));
    if i==1
        Cprob=importdata('morpho/NL000/prob1_10000.txt');
        % Clean
        [scores,pred]=max(Cprob,[],2);
        pred=2*pred-3;
        scores=scores.*pred;
        [tpr,tnr,info]=vl_roc(labels,scores);
        info.auc
        plot(1-tnr, tpr, 'color','k','linewidth',1.5)
        legends{end+1}='Clean';
        hold on
    end
   
    % Balanced (20%)
    [scores,pred]=max(bprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    info.auc
    plot(1-tnr, tpr,'--', 'color',col(i+mod(i,2)*2+1,:),'linewidth',1.5)
    legends{end+1}=sprintf('baseline 20%% N%d',i);
    hold on,
    
    % infogain balanced (20%)
    [scores,pred]=max(Hbprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    info.auc
    plot(1-tnr, tpr, 'color',col(i+mod(i,2)*2+1,:),'linewidth',1.5)
    hold on,
    legends{end+1}=sprintf('w/Q 20%% N%d',i);
end
legend(legends,'location','southwest','fontsize',12,'location','southeast')
print 'figures/tocrop/roc_infogain_balanced' '-dpdf'


%% ROC Curve - Tuning
close all; clc
figure,
legends={};
for i=1:nr
    bprob=importdata(sprintf('morpho/NL%s/bprob_10000.txt',NL{i}));
    Hbprob=importdata(sprintf('morpho/NL%s/Hbprob_10000.txt',NL{i}));
    if i==1
        Cprob=importdata('morpho/NL000/prob1_10000.txt');
        % Clean
        [scores,pred]=max(Cprob,[],2);
        pred=2*pred-3;
        scores=scores.*pred;
        [tpr,tnr,info]=vl_roc(labels,scores);
        info.auc
        plot(1-tnr, tpr, 'color','k','linewidth',1.5)
        legends{end+1}='Clean';
        hold on
    end
   
    % Balanced (20%)
    [scores,pred]=max(bprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    info.auc
    plot(1-tnr, tpr,'--', 'color',col(i+mod(i,2)*2+1,:),'linewidth',1.5)
    legends{end+1}=sprintf('baseline 20%% N%d',i);
    hold on,
    
    % Tuning balanced (20%)
    [scores,pred]=max(Hbprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    info.auc
    plot(1-tnr, tpr, 'color',col(i+mod(i,2)*2+1,:),'linewidth',1.5)
    hold on,
    legends{end+1}=sprintf('Tuned w/Q 20%% N%d',i);
end
legend(legends,'location','southwest','fontsize',12,'location','southeast')
print 'figures/tocrop/roc_tuning_balanced' '-dpdf'

%% ROC Curves - (2)

close all;
figure(3)
set(3,'units','normalized','position',[.1 .1 .1*nr .13*nr])
set(3,'PaperType','A4','PaperPositionMode','auto','paperOrientation','landscape')

for i=1:nr
    auc=zeros(1,6);
    bprob=importdata(sprintf('morpho/NL%s/bprob_10000.txt',NL{i}));
    prob=importdata(sprintf('morpho/NL%s/prob_10000.txt',NL{i}));
    Hprob=importdata(sprintf('morpho/NL%s/H2prob_10000.txt',NL{i}));
    Hbprob=importdata(sprintf('morpho/NL%s/Hbprob_10000.txt',NL{i}));
    Hbprob2=importdata(sprintf('morpho/NL%s/H2bprob_10000.txt',NL{i}));
    Cprob=importdata('morpho/NL000/prob1_10000.txt');
    vl_tightsubplot(4,i,'box','inner', 'margintop',.015, 'marginbottom', .04,'marginleft', .05,'marginright', .02) ;
    % Clean
    [scores,pred]=max(Cprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(1)=info.auc;
    plot(1-tnr, tpr, 'color','k','linewidth',1)
    hold on
    
    % Softmax
    [scores,pred]=max(prob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(2)=info.auc;
    plot(1-tnr, tpr,'color',col(1,:),'linewidth',1)
    hold on,
    %20%
    [scores,pred]=max(bprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(3)=info.auc;
    plot(1-tnr, tpr,'color',col(2,:),'linewidth',1)
    hold on,

    %Q tuning
    [scores,pred]=max(Hprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(4)=info.auc;
    plot(1-tnr, tpr, 'color',col(3,:),'linewidth',1)
    hold on,
    %20%
    [scores,pred]=max(Hbprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(5)=info.auc;
    plot(1-tnr, tpr, 'color',col(8,:),'linewidth',1)
    hold on,
    
    %w/Q directly
    [scores,pred]=max(Hbprob2,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(6)=info.auc;
    plot(1-tnr, tpr, 'color',col(5,:),'linewidth',1)
    
    legend({sprintf('Clean auc:%.3f',auc(1)),...
        sprintf('N%d -baseline 5%% auc:%.3f',i,auc(2)),...
        sprintf('N%d -baseline 20%% auc:%.3f',i,auc(3)),...
        sprintf('N%d -Tuning 5%% auc:%.3f',i,auc(4)),...
       sprintf('N%d -Tuning 20%% auc:%.3f',i,auc(5)),...
       sprintf('N%d -w/Q 20%% auc:%.3f',i,auc(6))},...
        'location','southeast','fontsize',12);
    

end
print 'figures/tocrop/roc_recap' '-dpdf'

%% 50%
%% Baseline (useless)
%% Accuracies
close all; clc;
NL={'TH10_30','TH20_20','TH30_30'};
nr=length(NL);
% Accuracies:
figure(1)
set(1,'units','normalized','position',[.1 .1 .55 .35])
set(1,'PaperType','A4','PaperPositionMode','auto')
for i=1:nr
    subplot(3,1,i)
    p1=importdata(sprintf('morpho/NL%s/p1.log.test',NL{i}));
    p2=importdata(sprintf('morpho/NL%s/fp1.log.test',NL{i}));
    plot(p1.data(:,1),p1.data(:,4),'col',col(1,:),'linewidth',2);
    hold on,
    plot(p2.data(:,1),p2.data(:,4),'col',col(6,:),'linewidth',2);
    ylabel('Accuracy')
    %ylim([.6 1])
    legend({'baseline 5%','baseline 50%'},'location','southwest','fontsize',10)
    title(sprintf('N%d',i))
    if i<3
        set(gca,'XTickLabel',{});
    end
end
xlabel('Iteration')
print 'figures/tocrop/acc_fifty' '-dpdf'

%% ROC Curve:
close all; clc
auc=zeros(1,nr+1);
figure,
legends={};
for i=1:nr
    bprob=importdata(sprintf('morpho/NL%s/fprob_10000.txt',NL{i}));
    prob=importdata(sprintf('morpho/NL%s/prob_10000.txt',NL{i}));
    if i==1
        Cprob=importdata('morpho/NL000/prob1_10000.txt');
        % Clean
        [scores,pred]=max(Cprob,[],2);
        pred=2*pred-3;
        scores=scores.*pred;
        [tpr,tnr,info]=vl_roc(labels,scores);
        info.auc
        plot(1-tnr, tpr, 'color','k','linewidth',1.5)
        legends{end+1}='clean';
        hold on
    end
    
    % Baseline 5%
    [scores,pred]=max(prob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    info.auc
    plot(1-tnr, tpr, 'color',col(i,:),'linewidth',1.5)
    legends{end+1}=sprintf('baseline 5%% N%d',i);
    hold on,
    
    % Balanced (50%)
    [scores,pred]=max(bprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    info.auc
    plot(1-tnr, tpr,'color',col(i,:),'linewidth',1.5)
    hold on,
    legends{end+1}=sprintf('baseline 50%% N%d',i);
    
    if i==nr
         % all 0
        [tpr,tnr,info]=vl_roc(labels,-1*ones(1,length(labels)));
        info.auc
        plot(1-tnr, tpr,'--','color','r','linewidth',1.5)
        legends{end+1}='All 0';
        hold on,
    end
end
legend(legends,'location','southwest','fontsize',12,'location','southeast')
print 'figures/tocrop/roc_baseline_fifty' '-dpdf'

%% Infogain
%% ROC Curves - (2)

close all;
figure(3)
figure(3)
set(3,'units','normalized','position',[.1 .1 .4 .52])
set(3,'PaperType','A4','PaperPositionMode','auto','paperOrientation','landscape')
auc=zeros(1,4);
for i=1:nr
    auc=zeros(1,6);
    prob=importdata(sprintf('morpho/NL%s/prob_10000.txt',NL{i}));
    bprob=importdata(sprintf('morpho/NL%s/H2bprob_10000.txt',NL{i}));
    fprob=importdata(sprintf('morpho/NL%s/Hfprob_10000.txt',NL{i}));
    Cprob=importdata('morpho/NL000/prob1_10000.txt');
    vl_tightsubplot(4,i,'box','inner', 'margintop',.015, 'marginbottom', .04,'marginleft', .05,'marginright', .02) ;
    % Clean
    [scores,pred]=max(Cprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(1)=info.auc;
    plot(1-tnr, tpr, 'color','k','linewidth',1)
    hold on
    
    % Softmax
    [scores,pred]=max(prob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(2)=info.auc;
    plot(1-tnr, tpr,'color',col(1,:),'linewidth',1)
    hold on,
    
    %20 w/Q%
    [scores,pred]=max(bprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(3)=info.auc;
    plot(1-tnr, tpr,'color',col(2,:),'linewidth',1)
    hold on,

    %50% w/Q
    [scores,pred]=max(fprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(4)=info.auc;
    plot(1-tnr, tpr, 'color',col(3,:),'linewidth',1)
    hold on,
    
    legend({sprintf('Clean auc:%.3f',auc(1)),...
        sprintf('N%d -baseline 5%% auc:%.3f',i,auc(2)),...
        sprintf('N%d -w/Q 20%% auc:%.3f',i,auc(3)),...
        sprintf('N%d -w/Q 50%% auc:%.3f',i,auc(4))},...
        'location','southeast','fontsize',12);
end
print 'figures/tocrop/roc_infogain_fifty' '-dpdf'

%% Fix the baseline:
%% ROC
close all,
f2prob=importdata(sprintf('morpho/NL%s/f2prob_10000.txt',NL{i}));
f3prob=importdata(sprintf('morpho/NL%s/f3prob_10000.txt',NL{i}));
Hrob=importdata(sprintf('morpho/NL%s/Hfprob_10000.txt',NL{i}));
Cprob=importdata('morpho/NL000/prob1_10000.txt');
figure,
% Clean
[scores,pred]=max(Cprob,[],2);
pred=2*pred-3;
scores=scores.*pred;
[tpr,tnr,info]=vl_roc(labels,scores);
info.auc
plot(1-tnr, tpr, 'color','k','linewidth',1.5)
hold on

% 50% baseline larger batch
[scores,pred]=max(f3prob,[],2);
pred=2*pred-3;
scores=scores.*pred;
[tpr,tnr,info]=vl_roc(labels,scores);
info.auc
plot(1-tnr, tpr, 'color',col(1,:),'linewidth',1.5)
hold on,

% 50% baseline with BN
[scores,pred]=max(f2prob,[],2);
pred=2*pred-3;
scores=scores.*pred;
[tpr,tnr,info]=vl_roc(labels,scores);
info.auc
plot(1-tnr, tpr, 'color',col(2,:),'linewidth',1.5)
hold on,
% 50% w/Q
[scores,pred]=max(Hprob,[],2);
pred=2*pred-3;
scores=scores.*pred;
[tpr,tnr,info]=vl_roc(labels,scores);
info.auc
plot(1-tnr, tpr, 'color',col(3,:),'linewidth',1.5)
legend({'Clean','baseline larger batch','Baseline + batch-normalization','w/Q'},...
    'location','southeast');
print 'figures/tocrop/fix_baseline_roc' '-dpdf'

%% Accuracies
close all;
figure(1)
set(1,'units','normalized','position',[.1 .1 .55 .25])
set(1,'PaperType','A4','PaperPositionMode','auto')
p0=importdata(sprintf('morpho/NL%s/fp1.log.test',NL{i}));
p1=importdata(sprintf('morpho/NL%s/fp2.log.test',NL{i}));
p2=importdata(sprintf('morpho/NL%s/Hfp1.log.test',NL{i}));
p3=importdata(sprintf('morpho/NL%s/fp3.log.test',NL{i}));
plot(p0.data(:,1),p0.data(:,4),'col','r','linewidth',1.2);
hold on,
plot(p1.data(:,1),p1.data(:,4),'col',col(1,:),'linewidth',1.2);
hold on,
plot(p2.data(:,1),p2.data(:,4),'col',col(2,:),'linewidth',1.2);
hold on,
plot(p3.data(:,1),p3.data(:,4),'col',col(3,:),'linewidth',1.2);
legend({'baseline','baseline+bn','w/Q','larger batch'},...
    'location','southwest');
print 'figures/tocrop/fix_baseline_acc' '-dpdf'
