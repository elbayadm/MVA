function [alpha, b, SV, perf,varargout] = ovrSVM(K, Ytr, C , coeffs , params)
% One vs rest -  Multiclass learning:
alpha   = {};
b       = {};
d       = {};
SV      = {};
perf    = {};
if params.MKL
    for i=0:9
        Y            = 2*(Ytr==i)-1;
        params.id    = sprintf('%d vs R',i);
        params.C     = C(i+1); 
        params.coeff = coeffs(i+1);
        [alpha{end+1},d{end+1}, b{end+1}, SV{end+1}, perf{end+1}] = SMO_MKL ( K, Y, params);
    end
    varargout{1} = d; 
else
    if iscell(K) % Custom coefficients per class
        disp('Computing custom kernel')
        for i=0:9
            dd  = params.d{i+1}; % array of length #Kernels            
            K_ = zeros(size(K{1}));
            for k=1: length(K)
                K_= K_ + dd(k)*K{k};
            end
            Y            = 2*(Ytr==i)-1;
            params.id    = sprintf('%d vs R',i);
            params.C     = C(i+1); 
            params.coeff = coeffs(i+1);
            [alpha{end+1}, b{end+1}, SV{end+1}, perf{end+1}] = svm_smo ( K_, Y, params); 
            assert(length(b)==i+1)
        end
    else
        for i=0:9
            Y            = 2*(Ytr==i)-1;
            params.id    = sprintf('%d vs R',i);
            params.C     = C(i+1); 
            params.coeff = coeffs(i+1);
            [alpha{end+1}, b{end+1}, SV{end+1}, perf{end+1}] = svm_smo ( K, Y, params); 
        end
    end
end
assert(length(b)==10)
assert(length(alpha)==10)
