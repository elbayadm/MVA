function [w, score] = logit_newton(X,Y,lambda)
% Solve the logistic regression optimization problem via Newton-Raphson
% Inputs: 
% - X the design matrix
% - Y the ground truth labels
% - lambda : the l2 regularization parameters
% Outputs:
% - the weights w
% - the objective function evolution score
if nargin<3
    lambda = 0;
end
%sigmoid function
sigmoid = @(x) 1./(1+exp(-x));
% Convergence tolerance
tol = .01;
d = size(X,1);
% initialize w with zeros
w = zeros(1,d);
% steps counter
k = 0;
score = [];
max_iters = 30;
while k<max_iters % continue until convergence criterion is met 
    k = k +1;
    w_prev = w;
    % update w (Newton-Raphson)
    G        = sigmoid(w_prev*X);
    score(end+1) = sum(log(1-G) + Y.*(w_prev*X)) - lambda*sum(w_prev.^2);
    % deerivatives:
    H        = -X*diag(G.*(1-G))*X' - 2*lambda*eye(d);
    J        = (Y-G)*X'-2*lambda*w_prev;
    % step
    w        = w_prev - J*pinv(H);       
    % convergence criterion
    if sqrt(sum((w-w_prev).^2))/sqrt(sum(w.^2))<tol
        break
    end
end
