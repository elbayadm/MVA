%% Location: path/to/matconvnet-1.0-beta16/examples
% disp(Train the models)
% clear; close all; clc;
% cnn_mnist_DLclass();
% clc;
% cnn_mnist_DLclass_v2();
% clc;
% cnn_mnist_DLclass_v3();
% clc;
% cnn_mnist_DLclass_v4();
% clc;
% cnn_mnist_DLclass_v5();
% clc;
% cnn_mnist_DLclass_v7();
% clc;
% cnn_mnist_DLclass_v8();
% clc;
% cnn_mnist_DLclass_v9();
% clc;
% cnn_mnist_DLclass_v10();
% clc;
% cnn_mnist_DLclass_v11();
%%---------------------------- Plot recap
colors=.8*hsv(13);
colors=colors(randperm(13,13),:);
models={'baseline','v2','v3','v4','v5','v6','v6bis','v7','v8','v9','v10','v11'};
%           1        2    3   4    5     6     7     8    9    10    11    12   
drop=[2 3 7];
figure,
fprintf('Model & Error & Objective & Train-Objective & Speed\\\\ \n');
for i=1:length(models)
	temp=load(sprintf('data/mnist-%s/net-epoch-100.mat',models{i}));
    if ~ismember(i,drop)
    plot(5:100,temp.info.val.error(1,5:100),'LineWidth',2,'color',colors(i,:));
    hold on,
    end
	err=temp.info.val.error(1,end)*100;
	obj=temp.info.val.objective(end);
    objt=temp.info.train.objective(end);
    sp=floor(temp.info.val.speed(end));
	fprintf('%s & %.2f\\%% & %.2e & %.2e & %d\\\\\n',models{i},err,obj,objt,sp);
end
xlabel('Training epoch')
ylabel('Validation error')
xlim([5,100]);
I=1:length(models);
I(ismember(I,drop))=[];
legend(models{I},'Location','Northeast');
print '-dpdf' 'recap';
