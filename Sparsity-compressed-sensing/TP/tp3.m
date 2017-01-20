%% Settings:
clearvars;close all;clc;
set(0,'DefaultAxesFontSize',12)
set(0,'DefaultTextFontSize',12)

% Loading toolboxes:
addpath(genpath('matlab-toolboxes/toolbox_signal'))
addpath(genpath('matlab-toolboxes/toolbox_general'))

%% ---------------------------------------
%% Inpainting using Sparse Regularization:
%% ---------------------------------------

n = 128;
name = 'lena';
f0 = load_image(name);
f0 = rescale(crop(f0,n));
figure(1); clf;
imageplot(f0, 'Image f_0');

% removed pixels:
rho = .7;
% Mask of random pixels:
Omega = zeros(n,n);
sel = randperm(n^2);
Omega(sel(1:round(rho*n^2))) = 1;
Phi = @(f,Omega)f.*(1-Omega);

% Damaged observations:
y = Phi(f0,Omega);
figure(1); clf;
imageplot(y, 'Observations y');

%% Soft Thresholding in a Basis
SoftThresh = @(x,T)x.*max( 0, 1-T./max(abs(x),1e-10) );
% 1D soft thresholding
figure(1); clf;
T = linspace(-1,1,1000);
plot( T, SoftThresh(T,.5) );
axis('equal');

% wavelet transform params
Jmax = log2(n)-1;
Jmin = Jmax-3;
options.ti = 0; % use orthogonality.
Psi = @(a)perform_wavelet_transf(a, Jmin, -1,options);
PsiS = @(f)perform_wavelet_transf(f, Jmin, +1,options);
SoftThreshPsi = @(f,T)Psi(SoftThresh(PsiS(f),T));
figure(1); clf;
imageplot( clamp(SoftThreshPsi(f0,.1)) );

%% Inpainting using Orthogonal Wavelet Sparsity
lambda = .03;
ProjC = @(f,Omega)Omega.*f + (1-Omega).*y;

%% Exo 1:
% initialization
fSpars = y;
E = [];
niter = 1000;
for i=1:niter
	% denoise
    fSpars = SoftThreshPsi( ProjC(fSpars,Omega), lambda );
    % track the energy
    fW = PsiS(fSpars);
    E(i) = 1/2 * norm(y-Phi(fSpars,Omega), 'fro')^2 + lambda * sum(abs(fW(:)));
end
figure(1); clf;
plot(E,'linewidth',2,'color','b');
axis('tight');
xlabel('Iteration')
ylabel('Energy')

figure(2);clf;
imageplot(clamp(fSpars));

%% Exo 2 : decaying lambda
niter = 1000;
lms = linspace(.03, 1e-5, niter);
err = [];
for i=1:niter
    fSpars = SoftThreshPsi( ProjC(fSpars,Omega), lms(i) );    
end
figure(1); clf;
imageplot(clamp(fSpars), ['Sparsity inpainting, SNR=' num2str(snr(f0,fSpars),3) 'dB']);


%% Inpainting using Translation Invariant Wavelet Sparsity
J = Jmax-Jmin+1;
u = [4^(-J) 4.^(-floor(J+2/3:-1/3:1)) ];
U = repmat( reshape(u,[1 1 length(u)]), [n n 1] );
lambda = .01;
options.ti = 1; % use translation invariance
Xi = @(a)perform_wavelet_transf(a, Jmin, -1,options);
PsiS = @(f)perform_wavelet_transf(f, Jmin, +1,options);
Psi = @(a)Xi(a./U);
tau = 1.9*min(u);

%% Exo 3
niter = 1000;
% initialization
a = U.*PsiS(fSpars);
E = [];
for i=1:niter
    % Gradient descent:
	fTI = Psi(a);
    d = y-Phi(fTI,Omega);
    E(i) = 1/2*norm(d , 'fro')^2 + lambda * sum( abs(a(:)) );   
    % update & soft thresholding
    a = SoftThresh( a + tau*PsiS(Phi(d,Omega)), lambda*tau );
end
figure(1); clf;
plot(E,'linewidth',2,'color','b'); 
axis('tight');

figure(2); clf;
fTI = Psi(a);
imageplot(clamp(fTI));

%% Exo 4 : decaying lambda
niter = 1000;
a = U.*PsiS(fSpars);
lms = linspace(.03, 1e-5, niter);
for i=1:niter
    fTI = Psi(a);
    d = y-Phi(fTI,Omega);
    % update & soft thresholding
    a = SoftThresh( a + tau * PsiS( Phi(d,Omega) ) , lms(i)*tau ) ;
end
figure(1); clf;
fTI = Psi(a);
imageplot(clamp(fTI), ['Sparsity inpainting TI, SNR=' num2str(snr(f0,fTI),3) 'dB']);

%% Inpainting using Iterative Hard Thresholding
HardThresh = @(x,t)x.*(abs(x)>t);
t = linspace(-1,1,1000);
plot( t, HardThresh(t,.5) );
axis('equal');

%% Exo 5 : decaying lambda
niter = 500;
lambda_list = linspace(1,0,niter);
% initialization
fHard = y;
for i=1:niter
	% Gradient descent
    fHard = Xi( HardThresh(PsiS(ProjC(fHard,Omega)), lambda_list(i)) );
end
figure(1); clf;
imageplot(clamp(fHard), ['Inpainting hard thresh., SNR=' num2str(snr(f0,fHard),3) 'dB']);

