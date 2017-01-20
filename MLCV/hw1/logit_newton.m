function [w, score] = logit_newton(X,Y,lambda)
% Solve the logistic regression optimization problem via Newton-Raphson
% Inputs: 
% - X the design matrix
% - Y the ground truth labels
% - lambda : the l2 regularization parameters
% Outputs:
% - the weights w
% - the objective function evolution score

%sigmoid function
sigmoid = @(x) 1./(1+exp(-x));
% Convergence tolerance
tol = .01;
[~,d] = size(X);
% initialize w with zeros
w = zeros(d,1);
% steps counter
k = 0;
score = [];
max_iters = 30;
while k<max_iters % continue until convergence criterion is met 
    k = k +1;
    w_prev = w;
    % update w (Newton-Raphson)
    G        = sigmoid(X*w_prev);
    
    score(end+1) = sum(log(1-G) + Y.*(X*w_prev)) - lambda*sum(w_prev.^2);
    fprintf('Iteration %2d | score %.3f\n',k,score(end))
    % deerivatives:
    H        = -X'*diag(G.*(1-G))*X - 2*lambda*eye(d);
    J        = X'*(Y-G)-2*lambda*w_prev;
    % step
    w        = w_prev - H\J;       
    % convergence criterion
    if sqrt(sum((w-w_prev).^2))/sqrt(sum(w.^2))<tol
        break
    end
end
