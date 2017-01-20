function X = scale(X)
for d=1:size(X,2)
    m = min(X(:,d));
    M = max(X(:,d));
    if (M-m)
        X(:,d) = (X(:,d)-m)/(M-m);
    else
        X(:,d) = 0;
    end
end
end
