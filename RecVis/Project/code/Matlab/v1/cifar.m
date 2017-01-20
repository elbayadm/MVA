%% Settings
clear, close all
addpath(genpath('Matlab/'))
addpath(genpath('logs/'))
addpath(genpath('features/'))
addpath(genpath('Confusion/'))

rand('seed',2);
col=.8*hsv(12);
col=col(randperm(12,12),:);
%% CIFAR:  trueQ vs learnt Q
close all
NL=[2 4 6]; %Choose 3
figure('units','normalized','position',[.1 .1 .4 .8])
for i=1:length(NL)
Q=load(sprintf('QNL0%d_5000',NL(i)));
Q=Q.Q;
trueQ=importdata(sprintf('cifar10/matrix_q0.%d.txt',NL(i)));
subplot(3,2,2*i-1)
imagesc(Q)
title('Learnt Q','fontSize',12)
subplot(3,2,2*i)
imagesc(trueQ)
title('True Q','fontSize',12)
txt=sprintf('Noise %d%% - diff=%.2f',NL(i)*10,norm(Q-trueQ,2));
annotation('textbox',[0.35 .89-0.31*(i-1) .1 .1],...
            'String',txt,'FontSize',14,'FontName','Helvetica','EdgeColor','none');   
end
set(gcf,'PaperType','A4','PaperPositionMode','auto')
print 'figures/tocrop/QQ' '-dpdf'
%%  Accuracies vs NoiseLevel(NL)
close all,
ACC={};
for i=1:9
    ACC{i}=[];
end
close all,
figure,
NL=[0 1 2 3 4 5 6 7];
iters=[0:1000:60000 60000:1000:65000 65000:1000:70000 70000:100:75000]; 
for i=1:length(NL)
    f1=importdata(sprintf('cifar10/NL0%d/f1.log.test',NL(i)));
    ACC{i}=[ACC{i}; f1.data(:,4)];
    f2=importdata(sprintf('cifar10/NL0%d/f2.log.test',NL(i)));
    ACC{i}=[ACC{i}; f2.data(:,4)];
    f3=importdata(sprintf('cifar10/NL0%d/f3.log.test',NL(i)));
    ACC{i}=[ACC{i}; f3.data(:,4)];
    f4=importdata(sprintf('cifar10/Qcifar10_NL0%d.log.test',NL(i)));
    ACC{i}=[ACC{i}; f4.data(:,4)];
    plot(iters,ACC{i},'color',col(i,:),'linewidth',2)
    hold on,
end
NL=10*NL;
line([60000 60000],[0 1],'color','k','linestyle','--')
line([65000 65000],[0 1],'color','k','linestyle','--')
line([70000 70000],[0 1],'color','k','linestyle','--')

legend(cellstr(num2str(NL', 'Noise level %d%%')),'location','southwest')
ylabel('Accuracy')
xlim([0 75000])
xlabel('Iteration')
grid on,
grid minor,
print 'figures/tocrop/training' '-dpdf'
%% Accuracies 2
NL=[1 2 3 4 5 6 7];
acc=zeros(1,length(NL));
acc_clean=zeros(1,length(NL));
acc_preQ=zeros(1,length(NL));

labels=importdata('cifar10/test_labels.txt');
labels=labels.data;
for i=1:length(NL)
trueQ=importdata(sprintf('cifar10/matrix_q0.%d.txt',NL(i)));
Q=load(sprintf('QNL0%d_5000',NL(i)));
Q=Q.Q;
wQ=importdata(sprintf('cifar10/NL0%d/prob_5000.txt',NL(i)));
ip=importdata(sprintf('cifar10/NL0%d/prob_70000.txt',NL(i)));
conf=importdata(sprintf('cifar10/NL0%d/conf_5000.txt',NL(i)));
wtQ=conf*pinv(trueQ'); %with true Q
[~,pred]=max(wQ,[],2);
[~,pred_clean]=max(wtQ,[],2);
[~,pred_preQ]=max(ip,[],2);
acc(i)=sum(pred-1==labels)/length(labels);
acc_clean(i)=sum(pred_clean-1==labels)/length(labels);
acc_preQ(i)=sum(pred_preQ-1==labels)/length(labels);
end
figure,
plot(10*NL,acc_clean,'-^','color',col(2,:),'linewidth',2)
hold on,
plot(10*NL,acc,'-x','color',col(5,:),'linewidth',2)
hold on,
plot(10*NL,acc_preQ,'-o','color',col(3,:),'linewidth',2)
xlabel('Noise level (%)')
ylabel('Accuracy')
legend({'with true Q','with learnt Q','baseline'},'Location','southwest')
grid on
grid minor
print 'figures/tocrop/acc' '-dpdf'