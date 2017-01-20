addpath('/usr/local/cellar/matconvnet/matlab')
addpath('../mlcv_toolbox')
vl_setupnn
clc;
%% Varying the learning rate and weight decay
classifier = 'lin';  % lin, log, svm
opts.dataDir = 'data';
opts.train.momentum = 0.0;
opts.train.numEpochs = 500 ;
opts.train.batchSize = 1000;
opts.train.gpus=[];
%opts.train.learningRate = 4e-6; %(LR)
%opts.train.learningRate = 1e-3; %(SVM)
%opts.train.learningRate = 8e-3; %(logit)
a = 6; b= 5;
alphas = logsample(1e-6,1e-4,a);
lambdas = [0 1e-4 1e-3 1e-2 1];
figure(1);clf;
for i=1:length(alphas)
    for j=1:length(lambdas)
        opts.train.learningRate = alphas(i);
        opts.train.weightDecay = lambdas(j);
        [net, info] = cnn_exp(classifier, opts);
        subplot(a,b,b*(i-1)+j)
        plot(info.train.objective,'linewidth',1); 
        title(sprintf('Loss %.1e \n \\alpha =%.1e \\lambda=%.1e\n#errors: %d',...
            info.train.objective(end),opts.train.learningRate ,opts.train.weightDecay, info.test.errors),'fontsize',7);
        drawnow
    end
end
%print '-dpdf' 'figures/lr'

%% varying the momentum:
classifier = 'svm';  % lin, log, svm
opts.dataDir = 'data';
opts.train.numEpochs = 500 ;
opts.train.batchSize = 1000;
opts.train.gpus=[];
%opts.train.learningRate = 4e-6; %(LR)
opts.train.learningRate = 1e-3; %(SVM)
%opts.train.learningRate = 8e-3; %(logit)
mus = [0. .5 .9];
figure(1);clf;
i=1;
for mu = mus
    opts.train.momentum = mu;
    [net, info] = cnn_exp(classifier, opts);
    subplot(1,3,i)
    plot(info.train.objective,'linewidth',2); 
    title(sprintf('Loss %.1e \n \\alpha =%.1e \\mu=%.1f\n#errors: %d',...
            info.train.objective(end),opts.train.learningRate ,opts.train.momentum, info.test.errors),'fontsize',12);
    drawnow
    i= i+1;
end
print '-dpdf' 'figures/svm_mom'

