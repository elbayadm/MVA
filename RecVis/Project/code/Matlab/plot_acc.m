col1=[202 12 12]/255; %red
col2=[77 168 95]/255; %green
col3=[22 15 240]/255; %Blue
col4=[235 175 9]/255; %Orange
set(0,'DefaultTextFontSize',13)

doprint=false;
%Clean
clean=importdata('../logs/cifar10_clean_10000.txt');
clean=clean.data;
%iter;lr;train_loss;test_loss,train_acc,test_acc

%Noise 
nl1=.1
noisy1=importdata('../logs/cifar10_n0.1_10000_p1.txt');
noisy1=noisy1.data;

%Noise with Q:
noisy1Q=importdata('../logs/cifar10_n0.1_10000_p2.txt');
noisy1Q=noisy1Q.data;

% Accuracy
figure('units','normalized','position',[.1 .1 .8 .3],'PaperUnits','normalized','PaperPosition',[.1 .1 1 .2])
plot(clean(:,1),1-clean(:,6),'color',col1)
hold on
plot(noisy1(:,1),1-noisy1(:,6),'color',col2)
legend({'Clean','NLevel=0.1'})
title('Test error - training size : 10000')
xlabel('Iteration')
ylabel('Error')
if doprint
	print('-dpng','../figures/p1_acc_clean_nodp')
end
% Q training
figure('units','normalized','position',[.1 .1 .8 .3],'PaperUnits','normalized','PaperPosition',[.1 .1 1 .2])
plot(noisy1Q(:,1),1-noisy1Q(:,6),'color',col3)
title('Q training')
xlabel('Iteration')
ylabel('Error')
% Losses
figure('units','normalized','position',[.1 .1 .8 .3],'PaperUnits','normalized','PaperPosition',[.1 .1 1 .2])
plot(clean(:,1),clean(:,3),'color',col1)
hold on,
plot(clean(:,1),clean(:,4),'color',col2)
legend({'train','test'},'Location','northeast')
title('Losses')
xlabel('Iteration')
ylabel('Loss')
if doprint
	print('-dpng','../figures/p1_loss_clean_nodp')
end