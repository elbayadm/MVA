function [acc] = ovoAccuracy(pairs,alpha,b,SV,idx,idxv, Kval,Ytr, Yval)
% One to one accuracy:
acc=zeros(10,10);
for j = 1:length(pairs)
    k=pairs(j,1); l=pairs(j,2);

    a_  = alpha{k+1}{l+1};
    b_  = b{k+1}{l+1};
    SV_ = SV{k+1}{l+1};

    % predict:
    I = idx{k+1}{l+1};
    J = idxv{k+1}{l+1};

    Y = Ytr(I);
    Y = 2*(Y==k)-1;
    margins=zeros(size(J));
    for i = 1:length(J)
        margins(i) = sum(a_(SV_).*Y(SV_).*Kval(I(SV_),J(i)))+b_;
    end
    pred = k * (margins > 0) + l * (margins < 0);
    acc(k+1,l+1) = sum(Yval(J) == pred)/length(J);
end
