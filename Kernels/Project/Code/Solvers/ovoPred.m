function [pred] = ovoPred( alpha, SV, b, Ytr, Ktest, pairs, idx)
% Full kernel:
votes=zeros(size(Ktest,2),length(pairs));
for i=1:length(pairs)
    k= pairs(i,1); l= pairs(i,2);
    a_  = alpha{k+1}{l+1};
    b_  = b{k+1}{l+1};
    SV_ = SV{k+1}{l+1};

    % predict:
    I = idx{k+1}{l+1};

    Y = Ytr(I);
    Y = 2*(Y==k)-1;
    margins=zeros(size(Ktest,2),1);
    for j = 1:length(margins)
        margins(j) = sum(a_(SV_).*Y(SV_).*Ktest(I(SV_),j))+b_;
    end
    votes(:,i) = k * (margins > 0) + l * (margins < 0);
    
end
pred= mode(votes,2);