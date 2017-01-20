%% Settings:
clearvars;close all;clc;
set(0,'DefaultAxesFontSize',12)
set(0,'DefaultTextFontSize',12)

% Loading toolboxes:
addpath(genpath('matlab-toolboxes/toolbox_signal'))
addpath(genpath('matlab-toolboxes/toolbox_general'))

%% --------------------------------
%% Primal-Dual Proximal Splitting 
%% --------------------------------
%% Inpainting Problem
name = 'lena';
n = 256;
f0 = load_image(name);
f0 = rescale(crop(f0,n));
figure(1); clf;
imageplot(f0);

% Random mask
rho = .8;
Lambda = rand(n,n)>rho;
Phi = @(f)f.*Lambda;

% The obseravations:
y = Phi(f0);
figure(1); clf;
imageplot(y);

%% Total Variation Regularization under Constraints
K  = @(f)grad(f);
KS = @(u)-div(u);
Amplitude = @(u)sqrt(sum(u.^2,3));
F = @(u)sum(sum(Amplitude(u)));

% Display the thresholding on the vertical component of the vector.
ProxF = @(u,lambda)max(0,1-lambda./repmat(Amplitude(u), [1 1 2])).*u;
t = -linspace(-2,2, 201);
[Y,X] = meshgrid(t,t);
U = cat(3,Y,X);
V = ProxF(U,1);
figure(1);clf;
surf(V(:,:,1));
colormap jet(256);
view(150,40);
axis('tight');
camlight; shading interp;

% Display this dual proximal on the vertical component of the vector.
ProxFS = @(y,sigma)y-sigma*ProxF(y/sigma,1/sigma);
V = ProxFS(U,1);
figure(2); clf;
surf(V(:,:,1));
colormap jet(256);
view(150,40);
axis('tight');
camlight; shading interp;

%% Primal-dual Total Variation Regularization Algorithm
%% Exo 1:
ProxG = @(f,tau)f + Phi(y - Phi(f));
L = 8;
sigma = 10;
tau = .9/(L*sigma);
theta = 1;


niter = 200;
E = []; 
% Initialization:
f = y;
g = K(y)*0;
f1 = f;
for i=1:niter    
    % update
    fold = f;
    g = ProxFS( g+sigma*K(f1), sigma);
    f = ProxG(  f-tau*KS(g), tau);
    f1 = f + theta * (f-fold);
    % track the energy
    E(i) = F(K(f));
end
figure(1); clf;
plot(E,'linewidth',2);
xlabel('iteration')
ylabel('Energy')
axis('tight');

figure(2);clf;
imageplot(f, ['SNR=' num2str(snr(f,f0),3) 'dB']);

%% Exo 2:
epsilon = 3.5;
%SNR=[];
%for epsilon =linspace(2,6,30);
    ProxG = @(f,tau) f + Phi(epsilon*(y - Phi(f))/norm(y-Phi(f)));
    L = 8;
    sigma = 10;
    tau = .9/(L*sigma);
    theta = 1;
    niter = 200;
    E = []; 
    % Initialization:
    f = y;
    g = K(y)*0;
    f1 = f;
    for i=1:niter    
        % update
        fold = f;
        g = ProxFS( g+sigma*K(f1), sigma);
        f = ProxG(  f-tau*KS(g), tau);
        f1 = f + theta * (f-fold);
        % track the energy
        E(i) = F(K(f));
    end
    figure(3); clf;
    plot(E,'linewidth',2);
    xlabel('iteration')
    ylabel('Energy')
    axis('tight');
    drawnow,

    figure(4);clf;
    imageplot(f, ['SNR=' num2str(snr(f,f0),3) 'dB']);
    drawnow,
%     fprintf('%.2f : %.3f\n',epsilon , snr(f,f0));
%     SNR(end+1)=snr(f,f0);
% end
% figure,
% eps= linspace(2,6,30);
% plot(linspace(2,6,30), SNR, '-xb')
% [v,bs]=max(SNR);
% eps(bs)

%% Inpainting Large Missing Regions
n = 64;
name = 'square';
f0 = load_image(name,n);
a = 4;
Lambda = ones(n);
Lambda(end/2-a:end/2+a,:) = 0;
Phi = @(f)f.*Lambda;
figure(1); clf;
imageplot(f0, 'Original', 1,2,1);
imageplot(Phi(f0), 'Damaged', 1,2,2);
%% Exo 3:
y = Phi(f0);
ProxG = @(f,tau)f + Phi(y - Phi(f));
niter = 600;
E = [];
f = y;
g = K(y)*0;
f1 = f;
plt = 1;
figure(1); clf;
for i=1:niter    
    % update
    fold = f;
    g = ProxFS( g+sigma*K(f1), sigma);
    f = ProxG(  f-tau*KS(g), tau);
    f1 = f + theta * (f-fold);
    % track the energy
    E(i) = F(K(f));
    if ~mod(i,150)
        imageplot(f, sprintf('Iteration %d',i), 2,2,plt);
        plt = plt + 1;
    end
end
