%% Settings:
clearvars;close all;clc;
set(0,'DefaultAxesFontSize',12)
set(0,'DefaultTextFontSize',12)

% Loading toolboxes:
addpath(genpath('matlab-toolboxes/toolbox_signal'))
addpath(genpath('matlab-toolboxes/toolbox_general'))

%% ----------------------------------------------
%% Image Deconvolution using Variational Method:
%% ----------------------------------------------
%% Image Blurring
n = 256;
% name = 'lena';
 name = 'mri';
% name = 'boat';
f0 = load_image(name);
f0 = rescale(crop(f0,n));

%% Kernel :
s=3;
x = [0:n/2-1, -n/2:-1];
[Y,X] = meshgrid(x,x);
h = exp( (-X.^2-Y.^2)/(2*s^2) );
h = h/sum(h(:));

hF = real(fft2(h));

figure(1);clf;
imageplot(fftshift(h), 'Filter', 1,2,1);
imageplot(fftshift(hF), 'Fourier transform', 1,2,2);

% denoising
Phi = @(x,h)real(ifft2(fft2(x).*fft2(h)));
y0 = Phi(f0,h);
figure(1);clf;
clf;
imageplot(f0, 'Image f0', 1,2,1);
imageplot(y0, 'Observation without noise', 1,2,2);

% blurring
sigma = .02;
y = y0 + randn(n)*sigma;
figure(1);clf;
imageplot(y0, 'Observation without noise', 1,2,1);
imageplot(clamp(y), 'Observation with noise', 1,2,2);

%% Deconvolution with L2 Regularization
yF = fft2(y);
lambda = 0.02;
fL2 = real( ifft2( yF .* hF ./ ( abs(hF).^2 + lambda) ) );
figure(1);clf;
imageplot(y, strcat(['Observation, SNR=' num2str(snr(f0,y),3) 'dB']), 1,2,1);
imageplot(clamp(fL2), strcat(['L2 deconvolution, SNR=' num2str(snr(f0,fL2),3) 'dB']), 1,2,2);

%% Exo 1:
ls = linspace(2e-3 , 3e-2,100);
SNR=[];
for lambda = ls
    fL2 = real( ifft2( yF .* hF ./ ( abs(hF).^2 + lambda) ) );
    SNR(end+1)=snr(f0,fL2);
end
figure(1);clf;
plot(ls,SNR,'b','linewidth',2)
xlabel('\lambda')
ylabel('SNR')
title('L2 Regularization')
axis('tight')

[~,best] = max(SNR);
lambda = ls(best);
fL2 = real( ifft2( yF .* hF ./ ( abs(hF).^2 + lambda) ) );
figure(2);clf;
imageplot(y, strcat(['Observation, SNR=' num2str(snr(f0,y),3) 'dB']), 1,2,1);
imageplot(clamp(fL2), strcat(['L2 deconvolution, SNR=' num2str(snr(f0,fL2),3) 'dB']), 1,2,2);

%% Deconvolution by Sobolev Regularization.
S = (X.^2 + Y.^2)*(2/n)^2;
lambda = 0.2;
fSob = real( ifft2( yF .* hF ./ ( abs(hF).^2 + lambda*S) ) );
figure(1);clf;
imageplot(y, strcat(['Observation, SNR=' num2str(snr(f0,y),3) 'dB']), 1,2,1);
imageplot(clamp(fSob), strcat(['Sobolev deconvolution, SNR=' num2str(snr(f0,fSob),3) 'dB']), 1,2,2);

%% Exo 2:
ls = linspace(2e-2 , .2 ,100);
SNR=[];
for lambda = ls
    fSob = real( ifft2( yF .* hF ./ ( abs(hF).^2 + lambda*S) ) );
    SNR(end+1)=snr(f0,fSob);
end
figure(1);clf;
plot(ls,SNR,'b','linewidth',2)
xlabel('\lambda')
ylabel('SNR')
title('Sobolev Regularization')
axis('tight')

[~,best] = max(SNR);
lambda = ls(best);
fSob = real( ifft2( yF .* hF ./ ( abs(hF).^2 + lambda*S) ) );
figure(2);clf;
imageplot(y, strcat(['Observation, SNR=' num2str(snr(f0,y),3) 'dB']), 1,2,1);
imageplot(clamp(fSob), strcat(['Sobolev deconvolution, SNR=' num2str(snr(f0,fSob),3) 'dB']), 1,2,2);

%% Deconvolution by Total Variation Regularization
%% Exo 3:
epsilon = 0.4*1e-2;                %  regularization parameter for the TV norm
lambda = 0.06;                     % deconvolution regularization parameter
tau = 1.9/(1+8*lambda/epsilon);    % step size
% Initialization:
fTV = y;
niter = 600;
E=[];
% Gradient descent
for it=1:niter
    % Compute the gradient:
    Gr = grad(fTV);
    d = sqrt( epsilon^2 + sum3(Gr.^2,3) );
    G = -div( Gr./repmat(d, [1 1 2])  );
    % Update:
    e = Phi(fTV,h)-y;
    fTV = fTV - tau*( Phi(e,h) + lambda*G);
    % Track the objective:
    E(it) = 1/2*norm(e,'fro')^2 + lambda*sum(d(:));
end
figure(1);clf;
plot(1:niter, E,'linewidth',2,'color','b')
xlabel('iteration')
ylabel('Energy')
axis('tight')

% Result:
figure(2);clf;
imageplot(clamp(fTV));

%% Exo 4:

epsilon = 0.4*1e-2;                %  regularization parameter for the TV norm
niter = 350;
ls = linspace(.4e-3, 7e-3 ,30);
SNR = [];
fTVbest = y;
for i = 1:length(ls)
    lambda = ls(i);
    fprintf('\nlambda[%d]= %.2e :',i,lambda);
    tau = 1.9/(1+8*lambda/epsilon);    % step size
    % Initialization:
    fTV = y;
    E=[];
    % Gradient descent
    for it=1:niter
        if ~mod(it,40)
            fprintf('.')
        end
        % Compute the gradient:
        Gr = grad(fTV);
        d = sqrt( epsilon^2 + sum3(Gr.^2,3) );
        G = -div( Gr./repmat(d, [1 1 2])  );
        % Update:
        e = Phi(fTV,h)-y;
        fTV = fTV - tau*( Phi(e,h) + lambda*G);
    end
    SNR(end+1)=snr(f0,fTV);
    if SNR(end)> snr(f0,fTVbest)
        fTVbest = fTV;
    end
end
figure(1);clf;
plot(ls, SNR,'linewidth',2,'color','b')
xlabel('\lambda')
ylabel('SNR')
title('Total Variation Regularization')
axis('tight')

figure(2);clf;
imageplot(y, strcat(['Observation, SNR=' num2str(snr(f0,y),3) 'dB']), 1,2,1);
imageplot(clamp(fTVbest), strcat(['TV deconvolution, SNR=' num2str(snr(f0,fTVbest),3) 'dB']), 1,2,2);

%% Comparison of Variational and Sparsity Methods

figure(1);clf;
imageplot(y, strcat(['Observation, SNR=' num2str(snr(f0,y),3) 'dB']), 2,2,1);
imageplot(clamp(fL2), strcat(['L2, SNR=' num2str(snr(f0,fL2),3) 'dB']), 2,2,2);
imageplot(clamp(fSob), strcat(['Sobolev, SNR=' num2str(snr(f0,fSob),3) 'dB']), 2,2,3);
imageplot(clamp(fTVbest), strcat(['TV, SNR=' num2str(snr(f0,fTVbest),3) 'dB']), 2,2,4);
% L2 and Sobolev regularization yield similar results
% and are both outperformed by the TV regularization method with a better edge recovery. 