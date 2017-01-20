function [K] = kernel(X1,X2, params)
% Compute kernel matrix between X1 and X2 
% X1: (n1,d)  matrix 
% X2: (n2,d)  matrix
% K: (n1,n2)  matrix
tiny = 1e-10;
switch params.id
    case 'linear'
        K = X1*X2';
        K = K/params.scale; 
        
    case 'poly'
        K = ((params.shift + X1*X2')/params.scale).^params.deg;

    case 'rbf'
        n1 = size(X1,1);
        n2 = size(X2,1);
        D  = repmat(sum(X1.^2,2),1,n2)...
            +repmat(sum(X2.^2,2)',n1,1)...
            - 2*X1*X2';
        %D = pdist2(X1,X2,'euclidean').^2;
        K = exp(-D/(2*params.sigma^2));
        clear D;
    
    case 'tanh'
        K = tanh(X1*X2'/params.scale);
    
    case 'inter'
        K = pdist2(X1, X2, @(x, y) sum(bsxfun(@min, x, y), 2));
        K = K/params.scale; 
    
    case 'expchi2'
        n1 = size(X1,1);
        n2 = size(X2,1);       
        K = zeros(n1,n2);
        for i=1:n1
            numer = bsxfun(@minus, X1(i,:), X2);
            numer = numer.^2;
            denom = bsxfun(@plus, X1(i,:), X2);
            K(i,:)= sum(numer./(denom+tiny),2);
        end
        K = exp(-params.gamma* K);
   
    case 'chi2'
        n1 = size(X1,1);
        n2 = size(X2,1);       
        K = zeros(n1,n2);
        for i=1:n1
            numer = bsxfun(@minus, X1(i,:), X2);
            numer = numer.^2;
            denom = bsxfun(@plus, X1(i,:), X2);
            K(i,:)=  - sum(numer./(denom+tiny),2);
        end
        K = 1+(K/params.scale); 

    otherwise
        error('Unknown kernel type')
                 
end

