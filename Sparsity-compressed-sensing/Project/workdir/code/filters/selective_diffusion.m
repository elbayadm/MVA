function [D, D0] = selective_diffusion(I0,width,sx,sv,samp,method)
% regularization coefficient
eta = .2;
alpha = sqrt(2);

 switch method
     case 'closest'
        D0 = closest_mode(I0,width,sx,sv,samp);
     case 'bilateral'
        D0 = bilateral(I0,width,sx,sv,samp);
 end
 change = inf;
 sigma = sx;
 D = D0;
 iter = 1;
 while change 
     % blur
     B = conv2(D,GaussianKernel(sigma),'same'); 
     Eu = conv2((D - I0).^2,GaussianKernel(eta*sigma),'same');
     Eb = conv2((B - I0).^2,GaussianKernel(eta*sigma),'same');
     R  = Eb./(Eu + 1e-20);
     D = B.*(R<.5)+ D.*(R>=1)+...
         (.5*(R-.5).*(D-B)+B).*(R>=.5).*(R<1);
     change = sum(sum(R<1));
     fprintf('Iteration %d : changed %d pixels\n',iter,change);
     sigma = alpha * sigma;
     iter = iter +1;
 end
 
 
function [G] = GaussianKernel(sigma)
width = 1 + floor(sigma);
% Gaussian kernel:
[gridX, gridY] = meshgrid( 0 : (2*width), 0 : (2*width));
gridX = gridX - width;
gridY = gridY - width;
gridRSquared = ( gridX .* gridX + gridY .* gridY);
G = exp(gridRSquared/2/sigma^2);
G = G./sum(G(:));
