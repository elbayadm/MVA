function [pred, votes] = ovrPredMKL( alpha,d, SV, b, Ytr, Ktest, varargin)
% one vs rest prediction with MKL
if nargin == 7
    scales = varargin{1};
end
votes=zeros(size(Ktest{1},2),10);
for i=0:9
    a_  = alpha{i+1};
    b_  = b{i+1};
    SV_ = SV{i+1};
    d_  = d{i+1};
    K   = zeros(size(Ktest{1}));
    for k = 1: length(d_)
        K  =  K + d_(k)*Ktest{k};
    end
    Y = 2*(Ytr==i)-1;
    for j = 1:size(K,2)
        if nargin == 6 % Without platt's scaling
            votes(j,i+1) = sum(a_(SV_).*Y(SV_).*K(SV_,j))+b_;
        else
            sigmoid = @ (x,s) 1./(1 + exp(-s(1) - s(2)* x));
            scales_l = scales{i+1}; 
            votes(j,i+1) = sigmoid(sum(a_(SV_).*Y(SV_).*K(SV_,j))+b_,scales_l);
        end 
    end    
end

[~, pred]= max(votes,[],2);
pred = pred-1;
end