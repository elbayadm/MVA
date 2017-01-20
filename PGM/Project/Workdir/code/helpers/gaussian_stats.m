function [prob, mu] = gaussian_stats(x,I,h)

K = length(x);
if(length(I)==K) 
    mu = h./I;
    x_mu = x - mu;
    prob = exp(-0.5*x_mu.^2*I')*sqrt(prod(I));
else
    K = K/2;
    det = I(1:K).*I(K+1:2*K)-I(2*K+1:end).^2; 
    mu = [I(K+1:2*K),I(1:K)].*h-repmat(I(2*K+1:3*K),1,2).*[h(K+1:2*K),h(1:K)];
    mu = mu./repmat(det,1,2);
    x_mu = x - mu;
    prob = exp(-0.5*x_mu.^2*I(1:2*K)' - (x_mu(1:K).*x_mu(K+1:2*K))*I(2*K+1:end)');
    prob = prob*sqrt(prod(det));
end