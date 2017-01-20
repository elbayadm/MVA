%% Settings
clear, close all, clc
addpath(genpath('Matlab/'))
addpath(genpath('logs/'))
addpath(genpath('features/'))
addpath(genpath('Confusion/'))
labels=importdata('morpho/data/test_labels.txt');
labels=labels.data;
rand('seed',2);
col=.8*hsv(12);
col=col(randperm(12,12),:);
NL=[1 2 3];
NLB=[1 2];
%% Morpho:  trueQ vs learnt Q
nrows=length(NL)+length(NLB);
figure('units','normalized','position',[.1 .1 .3 .2*nrows])
for i=1:length(NL)
Q=load(sprintf('MQNL00%d_2500',NL(i)));
Q=Q.Q;
trueQ=importdata(sprintf('morpho/matrix_q0.0%d.txt',NL(i)));
% Print table
fprintf('NL %d\n',NL(i));
fprintf('%.2f & %.2f & %.2f & %.2f\n',Q(1,1),Q(1,2),trueQ(1,1),trueQ(1,2));
fprintf('%.2f & %.2f & %.2f & %.2f\n',Q(2,1),Q(2,2),trueQ(2,1),trueQ(2,2));
subplot(nrows,2,2*i-1)
imagesc(Q)
title('Learnt Q','fontSize',12)
subplot(nrows,2,2*i)
imagesc(trueQ)
title('True Q','fontSize',12)
txt=sprintf('Noise %d%%(1) - diff=%.2f',NL(i),norm(Q-trueQ,2));
% annotation('textbox',[0.31 .88-0.23*(i-1) .1 .1],...
%             'String',txt,'FontSize',14,'FontName','Helvetica','EdgeColor','none');   
end

for i=1:length(NLB)
j=i+length(NL);
Q=load(sprintf('MQNLB0%d_2500',NLB(i)));
Q=Q.Q;
trueQ=importdata(sprintf('morpho/matrix_qb0.%d.txt',NLB(i)));
% Print table
fprintf('NLb %d\n',NLB(i));
fprintf('%.2f & %.2f & %.2f & %.2f\n',Q(1,1),Q(1,2),trueQ(1,1),trueQ(1,2));
fprintf('%.2f & %.2f & %.2f & %.2f\n',Q(2,1),Q(2,2),trueQ(2,1),trueQ(2,2));
subplot(nrows,2,2*j-1)
imagesc(Q)
title('Learnt Q','fontSize',12)
subplot(nrows,2,2*j)
imagesc(trueQ)
title('True Q','fontSize',12)
txt=sprintf('Noise %d%%(2) - diff=%.2f',...
    10*NLB(i),norm(Q-trueQ,2));
% annotation('textbox',[0.31 .9-0.23*(j-1) .1 .1],...
%             'String',txt,'FontSize',14,'FontName','Helvetica','EdgeColor','none');   
end
set(gcf,'PaperType','A4','PaperPositionMode','auto')
%print 'figures/tocrop/QQMB' '-dpdf'
%%  Accuracies vs NoiseLevel(NL)
close all,
ACC={};
for i=1:9
    ACC{i}=[];
end
close all,
figure,
iters=[100:100:10000 10000:100:12500]; 
for i=1:length(NL)
    p1=importdata(sprintf('morpho/NL00%d/p1.log.test',NL(i)));
    ACC{i}=[ACC{i}; p1.data(:,4)];
    p2=importdata(sprintf('morpho/NL00%d/p2.log.test',NL(i)));
    ACC{i}=[ACC{i}; p2.data(:,4)];
    plot(iters,ACC{i},'color',col(i,:),'linewidth',2)
    hold on,
end

iters=[100:100:10000 10000:100:12500]; 
for i=1:length(NLB)
    j=i+length(NL);
    p1=importdata(sprintf('morpho/NLB0%d/p1.log.test',NLB(i)));
    ACC{j}=[ACC{j}; p1.data(:,4)];
    p2=importdata(sprintf('morpho/NLB0%d/p2.log.test',NLB(i)));
    ACC{j}=[ACC{j}; p2.data(:,4)];
    plot(iters,ACC{j},'color',col(j,:),'linewidth',2)
    hold on,
end
line([1 12500],[1-sum(labels)/length(labels) 1-sum(labels)/length(labels)],'color','k')
legend([cellstr(num2str(NL', 'Noise level %d%%(1)'));...
    cellstr(num2str(10*NLB', 'Noise level %d%%(2)'));'All 0'],...
    'location','southwest')
line([10000 10000],[.65 1],'color','k','linestyle','--')
ylabel('Accuracy')
xlim([0 12500])
xlabel('Iteration')
%print 'figures/tocrop/trainingM' '-dpdf'

%% Accuracies 2
acc=zeros(1,length(NL)+length(NLB));
acc_clean=zeros(1,length(NL)+length(NLB));
acc_preQ=zeros(1,length(NL)+length(NLB));
for i=1:length(NL)
trueQ=importdata(sprintf('morpho/matrix_q0.0%d.txt',NL(i)));
Q=load(sprintf('MQNL00%d_2500',NL(i)));
Q=Q.Q;
wQ=importdata(sprintf('morpho/NL00%d/prob_2500.txt',NL(i)));
ip=importdata(sprintf('morpho/NL00%d/prob_10000.txt',NL(i)));
conf=importdata(sprintf('morpho/NL00%d/conf_2500.txt',NL(i)));
wtQ=conf*pinv(trueQ'); %with true Q
[~,pred]=max(wQ,[],2);
[~,pred_clean]=max(wtQ,[],2);
[~,pred_preQ]=max(ip,[],2);
acc(i)=sum(pred-1==labels)/length(labels);
acc_clean(i)=sum(pred_clean-1==labels)/length(labels);
acc_preQ(i)=sum(pred_preQ-1==labels)/length(labels);
end

for i=1:length(NLB)
j=i+length(NL);
trueQ=importdata(sprintf('morpho/matrix_qb0.%d.txt',NLB(i)));
Q=load(sprintf('MQNLB0%d_2500',NLB(i)));
Q=Q.Q;
wQ=importdata(sprintf('morpho/NLB0%d/prob_2500.txt',NLB(i)));
ip=importdata(sprintf('morpho/NLB0%d/prob_10000.txt',NLB(i)));
conf=importdata(sprintf('morpho/NLB0%d/conf_2500.txt',NLB(i)));
wtQ=conf*pinv(trueQ'); %with true Q
[~,pred]=max(wQ,[],2);
[~,pred_clean]=max(wtQ,[],2);
[~,pred_preQ]=max(ip,[],2);
acc(j)=sum(pred-1==labels)/length(labels);
acc_clean(j)=sum(pred_clean-1==labels)/length(labels);
acc_preQ(j)=sum(pred_preQ-1==labels)/length(labels);
end

figure,
plot([NL 10*NLB],acc_clean,'-^','color',col(2,:),'linewidth',2)
hold on,
plot([NL 10*NLB],acc,'-x','color',col(5,:),'linewidth',2)
hold on,
plot([NL 10*NLB],acc_preQ,'-o','color',col(3,:),'linewidth',2)

line([0 20],[1-sum(labels)/length(labels) 1-sum(labels)/length(labels)],'color','k','linestyle','--')

xlabel('Noise level (%)')
ylabel('Accuracy')
legend({'with true Q','with learnt Q','baseline','All 0'},'Location','southwest')
grid on
grid minor
%print 'figures/tocrop/accM' '-dpdf'