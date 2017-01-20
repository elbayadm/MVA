function [scales] = plattMKL(alpha,d,SV,b,Ytr,K,perf)
% Platt's scaling:
[~, votesT] = ovrPredMKL( alpha,d, SV, b, Ytr, K);
scales = {};

for label = 0:9
    np = perf{label+1}.pos;
    nn = perf{label+1}.ns - np;
    tp = (np + 1) / (np + 2);
    tn = 1/(nn + 2);
    Y  = tp *(Ytr == label) + tn *(Ytr ~= label);
    scales{end+1} = train_logit( votesT(:,label+1), Y);
end