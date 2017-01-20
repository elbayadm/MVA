%% Settings
clear; close all; clc;
warning 'off'

addpath(genpath('Matlab/'))
addpath(genpath('logs/'))
addpath(genpath('features/'))
addpath('Confusion/')
addpath('data/morpho_QQ/original')

set(0, 'DefaultAxesFontName', 'Helvetica')
set(0,'defaultTextFontName', 'Helvetica')
set(0,'DefaultAxesFontSize',10)
set(0,'DefaultTextFontSize',10)

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

%% White noise
%% ACC
close all; clc;
NL={'001','002','003','B01','B015','B02'};
nr=length(NL);
% Accuracies:
figure(1)
set(1,'units','normalized','position',[.1 .1 .4 .2*nr])
set(1,'PaperType','A4','PaperPositionMode','auto')
for i=1:nr
    subplot(nr,1,i)
    p1=importdata(sprintf('morpho/NL%s/p1.log.test',NL{i}));
    p2=importdata(sprintf('morpho/NL%s/Hp1.log.test',NL{i}));
    plot(p1.data(:,1),p1.data(:,4),'col',col(1,:),'linewidth',1.5);
    hold on,
    plot(p2.data(:,1),p2.data(:,4),'col',col(6,:),'linewidth',1.5);
    ylabel('Accuracy')
    ylim([.8 1])
    legend({'baseline','w/Q'},'location','southwest','FontSize',10)
    title(sprintf('N%d',i))
    if i<nr
        set(gca,'XTickLabel',{});
    end
end
xlabel('Iteration')
%print 'figures/tocrop/acc' '-dpdf'
%% ROC Curve:
close all; clc
%figure,
legends={};
auc=[];
auc_Q=[];
NL={'001','002','003','B01','B015','B02'};
nr=length(NL);
for i=1:nr
    Hprob=importdata(sprintf('morpho/NL%s/Hprob_10000.txt',NL{i}));
    prob=importdata(sprintf('morpho/NL%s/prob_10000.txt',NL{i}));
    if i==1
        Cprob=importdata('morpho/NL000/prob1_10000.txt');
        % Clean
        [scores,pred]=max(Cprob,[],2);
        pred=2*pred-3;
        scores=scores.*pred;
%         [tpr,tnr,info]=vl_roc(labels,scores);
%         auc(end+1)=info.auc;
        [PREC, TPR, FPR, THRESH]=...
            prec_rec(scores,labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col',[col(i,:) ,'line', '-']);
        %plot(TPR, PREC, 'color','k','linewidth',1.5)
        legends{end+1}='clean';
        %hold on
    end
    
    % Softmax
    [scores,pred]=max(prob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(end+1)=info.auc;
    [PREC, TPR, FPR, THRESH]=...
       prec_rec(scores,labels,'numThresh',8000,'plotROC',0,'plotPR',1,'col',col(i,:),'line','-','holdFigure',1);
    %plot(1-tnr, tpr,'--', 'color',col(i,:),'linewidth',1.5)
    %plot(TPR,PREC,'--', 'color',col(i,:),'linewidth',1.5)
    legends{end+1}=sprintf('baseline [q0=%.2f, q1=%.2f]',q0(i),q1(i));

    %Q injected
    [scores,pred]=max(Hprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    [PREC, TPR, FPR, THRESH]=...
        prec_rec(scores,labels,'numThresh',8000,'plotROC',0,'plotPR',1,'style',col(i,:),'line','--','holdFigure',1);
    auc_Q(end+1)=info.auc;
    %plot(1-tnr, tpr, 'color',col(i,:),'linewidth',1.5)
    %plot(TPR,PREC, 'color',col(i,:),'linewidth',1.5)
    %hold on,
    legends{end+1}=sprintf('with Q',i);
end
%legend(legends,'location','southwest','fontsize',12,'location','southeast')
%grid on,
%axis square
%print 'figures/rapport/tocrop/pr' '-dpdf'

%% Image dependent
%%
close all; clc;
NL={'TH10_30','TH20_20','TH30_30','TH40_40'};
nr=length(NL);
% Accuracies:
figure(1)
set(1,'units','normalized','position',[.1 .1 .4 .2*nr])
set(1,'PaperType','A4','PaperPositionMode','auto')
for i=1:nr
    subplot(nr,1,i)
    p1=importdata(sprintf('morpho/NL%s/p1.log.test',NL{i}));
    p2=importdata(sprintf('morpho/NL%s/Hp1.log.test',NL{i}));
    plot(p1.data(:,1),p1.data(:,4),'col',col(1,:),'linewidth',1.5);
    hold on,
    plot(p2.data(:,1),p2.data(:,4),'col',col(6,:),'linewidth',1.5);
    ylabel('Accuracy')
    ylim([.8 1])
    legend({'baseline','w/Q'},'location','southwest')
    title(sprintf('N%d',i))
    if i<nr
        set(gca,'XTickLabel',{});
    end
end
xlabel('Iteration')
print 'figures/tocrop/acc_im' '-dpdf'
%% ROC Curve:
close all; clc
NL={'TH10_30','TH20_20','TH30_30','TH40_40'};
nr=length(NL);
auc=[];
auc_Q=[];
figure,
legends={};
for i=1:nr
    Hprob=importdata(sprintf('morpho/NL%s/Hprob_10000.txt',NL{i}));
    prob=importdata(sprintf('morpho/NL%s/prob_10000.txt',NL{i}));
    if i==1
        Cprob=importdata('morpho/NL000/prob1_10000.txt');
        % Clean
        [scores,pred]=max(Cprob,[],2);
        pred=2*pred-3;
        scores=scores.*pred;
        [tpr,tnr,info]=vl_roc(labels,scores);
        auc(end+1)=info.auc;
        plot(1-tnr, tpr, 'color','k','linewidth',1.5)
        legends{end+1}='clean';
        hold on
    end
    
    % Softmax
    [scores,pred]=max(prob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(end+1)=info.auc;
    plot(1-tnr, tpr, '--','color',col(i,:),'linewidth',1.5)
    legends{end+1}=sprintf('baseline (%d)',i);
    hold on,

    %Q injected
    [scores,pred]=max(Hprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc_Q(end+1)=info.auc;
    plot(1-tnr, tpr, 'color',col(i,:),'linewidth',1.5)
    hold on,
    legends{end+1}=sprintf('with Q (%d)',i);
end
xlim([0 .5])
ylim([.5 1])
grid on
axis square
legend(legends,'location','southwest','fontsize',12,'location','southeast')
print 'figures/rapport/tocrop/iroc_zoom' '-dpdf'

%% ROC Curves - (2)
close all;
figure(3)
set(3,'units','normalized','position',[.1 .1 .1*nr .13*nr])
set(3,'PaperType','A4','PaperPositionMode','auto','paperOrientation','landscape')
auc=zeros(1,4);
for i=1:nr
    Hprob=importdata(sprintf('morpho/NL%s/Hprob_10000.txt',NL{i}));
    prob=importdata(sprintf('morpho/NL%s/prob_10000.txt',NL{i}));
    Cprob=importdata('morpho/NL000/prob2_10000.txt');
    %subplot(2,2,i)
    vl_tightsubplot(4,i,'box','inner', 'margintop',.015, 'marginbottom', .04,'marginleft', .04,'marginright', .02) ;
    % Clean
    [scores,pred]=max(Cprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(1)=info.auc;
    plot(1-tnr, tpr, 'color','k')
    hold on
    
    % Softmax
    [scores,pred]=max(prob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(2)=info.auc;
    plot(1-tnr, tpr,'color','r')
    hold on,

    %Q injected
    [scores,pred]=max(Hprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(3)=info.auc;
    plot(1-tnr, tpr, 'color','b')
    hold on,
    
   
    
    legend({sprintf('Clean auc:%.3f',auc(1)),...
        sprintf('N%d -baseline auc:%.3f',i,auc(2)),...
        sprintf('N%d -w/Q  auc:%.3f',i,auc(3))},...
        'location','southeast');
      
end
print 'figures/tocrop/roc_im2' '-dpdf'

%% Clean databases:
set(0,'DefaultAxesFontSize',14)
set(0,'DefaultTextFontSize',14)
C1=importdata('morpho/NL000/prob1_10000.txt');
C2=importdata('morpho/NL000/prob2_10000.txt');
HC2=importdata('morpho/NL000/Hprob_10000.txt');

figure,
%1
[scores,pred]=max(C1,[],2);
pred=2*pred-3;
scores=scores.*pred;
[tpr,tnr,info]=vl_roc(labels,scores);
auc(i)=info.auc;
plot(1-tnr, tpr, 'color',col(1,:),'linewidth',1.5)
hold on
%2
[scores,pred]=max(C2,[],2);
pred=2*pred-3;
scores=scores.*pred;
[tpr,tnr,info]=vl_roc(labels,scores);
auc(i)=info.auc;
plot(1-tnr, tpr, 'color',col(2,:),'linewidth',1.5)
hold on
%3
[scores,pred]=max(HC2,[],2);
pred=2*pred-3;
scores=scores.*pred;
[tpr,tnr,info]=vl_roc(labels,scores);
auc(i)=info.auc;
plot(1-tnr, tpr, 'color',col(3,:),'linewidth',1.5)

legend({'C1','C2','C2 w/Q'},'location','southeast')
print 'figures/rapport/tocrop/dclean' '-dpdf'

%% Tuning
%% Roc (1)
close all;clc;
legends={};
% NL={'TH10_30','TH20_20','TH30_30','TH40_40'};
% nr=length(NL);
figure(1),
for i=1:nr
    Hprob=importdata(sprintf('morpho/NL%s/H2prob_10000.txt',NL{i}));
    prob=importdata(sprintf('morpho/NL%s/prob_10000.txt',NL{i}));
   if i==1
        Cprob=importdata('morpho/NL000/prob2_10000.txt');
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
    
%     % Softmax
%     [scores,pred]=max(prob,[],2);
%     pred=2*pred-3;
%     scores=scores.*pred;
%     [tpr,tnr,info]=vl_roc(labels,scores);
%     info.auc
%     plot(1-tnr, tpr,'--', 'color',col(i,:),'linewidth',1.5)
%     legends{end+1}=sprintf('baseline N%d',i);
%     hold on,

    %Q tuning
    [scores,pred]=max(Hprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    info.auc
    plot(1-tnr, tpr, 'color',col(i,:),'linewidth',1.5)
    hold on,
    legends{end+1}=sprintf('Tuned w/Q N%d',i);
    
end
legend(legends,'location','southwest','fontsize',12,'location','southeast')
print 'figures/tocrop/roc_tuning_bis' '-dpdf'

%% ROC Curves - (2)
close all;
figure(3)
set(3,'units','normalized','position',[.1 .1 .1*nr .13*nr])
set(3,'PaperType','A4','PaperPositionMode','auto','paperOrientation','landscape')
auc=zeros(1,3);
for i=1:nr
    Hprob=importdata(sprintf('morpho/NL%s/H2prob_10000.txt',NL{i}));
    prob=importdata(sprintf('morpho/NL%s/prob_10000.txt',NL{i}));
    Cprob=importdata('morpho/NL000/prob2_10000.txt');
    %subplot(1,3,i)
    vl_tightsubplot(4,i,'box','inner', 'margintop',.015, 'marginbottom', .04,'marginleft', .04,'marginright', .02) ;
    % Clean
    [scores,pred]=max(Cprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(1)=info.auc;
    plot(1-tnr, tpr, 'color','k')
    hold on
    
    % Softmax
    [scores,pred]=max(prob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(2)=info.auc;
    plot(1-tnr, tpr,'color','r')
    hold on,

    %Q injected
    [scores,pred]=max(Hprob,[],2);
    pred=2*pred-3;
    scores=scores.*pred;
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(3)=info.auc;
    plot(1-tnr, tpr, 'color','b')
    hold on,
    legend({sprintf('Clean auc:%.3f',auc(1)),...
        sprintf('N%d -baseline auc:%.3f',i,auc(2)),...
        sprintf('N%d -tuned w/Q  auc:%.3f',i,auc(3))},...
        'location','southeast');
end
print 'figures/tocrop/roc_tuning2' '-dpdf'

%% Accuracies:
close all;
figure(3)
legends={};
h={};
set(3,'units','normalized','position',[.1 .3 .7 .25])
set(3,'PaperType','A4','PaperPositionMode','auto','paperOrientation','landscape')
p1=importdata('morpho/NL000/p1.log.test');
%p2=importdata('morpho/NL000/Hp1.log.test'); 
h{end+1}=plot(p1.data(:,1),p1.data(:,4),'col','k','linewidth',1);
hold on,
%plot(10000+p2.data(:,1),p2.data(:,4),'col','k','linewidth',1);
hold on,
legends{end+1}='clean';
for i=1:nr
    p1=importdata(sprintf('morpho/NL%s/p1.log.test',NL{i}));
    p2=importdata(sprintf('morpho/NL%s/Hp2.log.test',NL{i}));
    h{end+1}=plot(p1.data(:,1),p1.data(:,4),'col',col(i+mod(i,2)*2+1,:),'linewidth',1);
    hold on,
    plot(10000+p2.data(:,1),p2.data(:,4),'col',col(i+mod(i,2)*2+1,:),'linewidth',1);
    legends{end+1}=sprintf('Tuning NL%d',i);
end
line([10000 10000],[.8 1],'color','k','linestyle','--')
ylabel('Accuracy')
ylim([.8 1])
xlabel('Iteration')
legend([h{:}],legends,'location','southeast')
print 'figures/tocrop/acc_tuning' '-dpdf'


%% Benchmark
% B histograms:
close all;
BNO=log(importdata('BNO.txt'));
BYES=log(importdata('BYES.txt'));
BNO=exp(BNO);
BYES=exp(BYES);
xbins=min(BNO):(max(BNO)-min(BNO))/40:max(BNO);
figure(1)
set(1,'units','normalized','position',[.1 .1 .7 .5])
set(1,'PaperType','A4','PaperPositionMode','auto','paperOrientation','landscape')

h0=histogram(BNO,xbins);
xlabel('score')
set(h0,'FaceColor',col(4,:),'facealpha',.75)
hold on,
h1=histogram(BYES,xbins);
set(h1,'FaceColor',col(10,:),'facealpha',.75);
legend({'Class 0','Class 1'})
xlim([0,160])
line([10 10],[0 3700],'color','k','linestyle','--','linewidth',1)
line([30 30],[0 3700],'color','k','linestyle','--','linewidth',1)
line([20 20],[0 3700],'color','k','linestyle','--','linewidth',1)
line([40 40],[0 3700],'color','k','linestyle','--','linewidth',1)
data=[BYES;BNO];
annotation('textbox',[0.125 .82 .1 .1],...
            'String',...
            sprintf('%.2f%%',sum(data<10)/length(data)*100),...
            'FontSize',12,'EdgeColor','none');   
annotation('textbox',[0.175 .82 .1 .1],...
            'String',...
            sprintf('%.2f%%',sum((data<20).*(data>10))/length(data)*100),...
            'FontSize',12,'EdgeColor','none');  
annotation('textbox',[0.225 .82 .1 .1],...
            'String',...
            sprintf('%.2f%%',sum((data<30).*(data>20))/length(data)*100),...
            'FontSize',12,'EdgeColor','none');
annotation('textbox',[0.277 .82 .1 .1],...
            'String',...
            sprintf('%.2f%%',sum((data<40).*(data>30))/length(data)*100),...
            'FontSize',12,'EdgeColor','none');
annotation('textbox',[0.5 .82 .1 .1],...
            'String',...
            sprintf('%.2f%%',sum(data>40)/length(data)*100),...
            'FontSize',12,'EdgeColor','none');
        
print 'figures/tocrop/benchmark' '-dpdf'

%% HIST
full_train=importdata('full_train.txt');
full_train_scores=full_train.data;
close all
figure(2)
set(2,'units','normalized','position',[.1 .1 .7 .5])
set(2,'PaperType','A4','PaperPositionMode','auto','paperOrientation','landscape')
h1=histogram(full_train_scores,xbins);
set(h1,'FaceColor',[.6 .6 .6],'facealpha',.75);
xlabel('score')
xlim([0,160])
line([10 10],[0 1.9e5],'color','k','linestyle','--','linewidth',1)
line([30 30],[0 1.9e5],'color','k','linestyle','--','linewidth',1)
line([20 20],[0 1.9e5],'color','k','linestyle','--','linewidth',1)
line([40 40],[0 1.9e5],'color','k','linestyle','--','linewidth',1)
data=full_train_scores;
annotation('textbox',[0.125 .82 .1 .1],...
            'String',...
            sprintf('%.2f%%',sum(data<10)/length(data)*100),...
            'FontSize',12,'EdgeColor','none');   
annotation('textbox',[0.175 .82 .1 .1],...
            'String',...
            sprintf('%.2f%%',sum((data<20).*(data>10))/length(data)*100),...
            'FontSize',12,'EdgeColor','none');  
annotation('textbox',[0.225 .82 .1 .1],...
            'String',...
            sprintf('%.2f%%',sum((data<30).*(data>20))/length(data)*100),...
            'FontSize',12,'EdgeColor','none');
annotation('textbox',[0.277 .82 .1 .1],...
            'String',...
            sprintf('%.2f%%',sum((data<40).*(data>30))/length(data)*100),...
            'FontSize',12,'EdgeColor','none');
annotation('textbox',[0.5 .82 .1 .1],...
            'String',...
            sprintf('%.2f%%',sum(data>40)/length(data)*100),...
            'FontSize',12,'EdgeColor','none');
        
print 'figures/tocrop/dataset' '-dpdf'

%%
l=importdata('full_train.txt');
%%
l=l.textdata;
%%
file_ = fopen('data/morpho_QQ/images.txt','wt');
for row = 1:length(l)
   fprintf(file_,'%s -1\n',l{row});
end
fclose(file_);