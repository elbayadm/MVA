function [w] = train_logit( X, gt)
% Logistic regression 
sigmoid = @(x) 1./(1+exp(-x));
X = [ones(size(X,1),1) X];
initial_w = [1 1]';
maxiter = 10;
tol = 1e-4;
it = 0;
diff = 1;
while (diff > tol) && (it < maxiter)
    it = it + 1;
    y = X*initial_w;
    y = sigmoid(y);
	error_grad = X'*(y - gt);
    RC = y.*(1-y);
    R = spdiags( RC, 0, numel(RC), numel(RC) );
    H = X'*R*X;
    Hinv = pinv(H);
    w_new = initial_w - (Hinv*error_grad);
    diff = norm(w_new - initial_w);
    %fprintf('[it %d] diff = %.2e \n',it,diff);
    initial_w = w_new;

end

w = w_new;

end