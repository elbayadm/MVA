function [pairs, idx, alpha, b, SV, perf] = ovoSVM(K, Ytr,C,params)
% One vs One: Multiclass:
pairs = nchoosek(0:9,2);
% Indices per class - Training & Validation
idx = {};
for i=1:length(pairs)
    idx{pairs(i,1)+1}{pairs(i,2)+1} = union(find(Ytr==pairs(i,1)),find(Ytr==pairs(i,2)));
end
alpha   = {};
b       = {};
SV      = {};
perf    = {};
for i=1:length(pairs)
    
    I = idx{pairs(i,1)+1}{pairs(i,2)+1};
    Y = Ytr(I);
    Y = 2*(Y==pairs(i,1))-1;
    params.C     = C; 
    params.coeff = 0;
    params.id = sprintf('%d vs %d',pairs(i,:));
    [alpha{pairs(i,1)+1}{pairs(i,2)+1},...
     b{pairs(i,1)+1}{pairs(i,2)+1},...
     SV{pairs(i,1)+1}{pairs(i,2)+1},...
     perf{pairs(i,1)+1}{pairs(i,2)+1}] = svm_smo ( K(I,I), Y, params); 
end