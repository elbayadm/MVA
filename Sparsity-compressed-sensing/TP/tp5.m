%% Settings:
clearvars;close all;clc;
set(0,'DefaultAxesFontSize',12)
set(0,'DefaultTextFontSize',12)

% Loading toolboxes:
addpath(genpath('matlab-toolboxes/toolbox_signal'))
addpath(genpath('matlab-toolboxes/toolbox_general'))

%% -------------------------------------
%% Douglas Rachford Proximal Splitting 
%% -------------------------------------
set_rand_seeds(123456,123456);
n = 400;
p = round(n/4);
% Random Gaussian measurement matrix A
A = randn(p,n) / sqrt(p);
% Sparsity 
s = 17;
sel = randperm(n);
x0 = zeros(n,1);
x0(sel(1:s))=1;
y = A*x0;

%% Compressed Sensing Recovery using DR
% l1 prox
proxG = @(x,gamma)max(0,1-gamma./max(1e-15,abs(x))).*x;
t = linspace(-1,1);
figure(1); clf;
plot(t, proxG(t,.3));
axis('equal');
% H prox
pA = A'*(A*A')^(-1);
proxF = @(x,y)x + pA*(y-A*x);
mu = 1;
gamma = 1;
% rprox 
rproxG = @(x,tau)2*proxG(x,tau)-x;
rproxF = @(x,y)2*proxF(x,y)-x;
%% Exo 1:
niter = 500;
lun = [];
tx = zeros(n,1);
for i=1:niter
    tx = (1-mu/2)*tx + mu/2*rproxG(rproxF(tx,y),gamma);
    x = proxF(tx,y);
    lun(i) = norm(x,1);
end
figure(1);clf;
plot(lun,'linewidth',2,'color','b');
axis tight;
% in log scale:
figure(2); clf;
plot(log10(lun(1:end/2)-lun(end)),'linewidth',2,'color','b');
axis('tight');

%% Original and recovered signals:
figure(1); clf;
subplot(2,1,1);
plot_sparse_diracs(x0);
set_graphic_sizes([], 15);
title('Original Signal');
subplot(2,1,2);
plot_sparse_diracs(x);
set_graphic_sizes([], 15);
title('Recovered by L1 minimization');

%% Exo 2 : Recovery of a less sparse signal
figure(1); clf;
set(1,'units','normalized','position',[.1 .1 .8 .4])
SS =[15 20 25 30];
for j = 1:length(SS)
    s = SS(j); 
    sel = randperm(n);
    x0 = zeros(n,1);
    x0(sel(1:s))=1;
    y = A*x0;

    tx = zeros(n,1);
    niter = 400;
    lun = [];
    for i=1:niter
        tx = (1-mu/2)*tx + mu/2*rproxG( rproxF(tx,y),gamma );
    end
    x = proxF(tx,y);
    subplot(2,4,j);
    plot_sparse_diracs(x0);
    title(sprintf('Original (s=%d)',s),'fontsize',12);
    subplot(2,4,4+j);
    plot_sparse_diracs(x);
    title('Recovered','fontsize',12);
    drawnow,
end
% When the original signal doesn't have a sparse representation (s>25) 
% the compressed sensing recovery with l1 normalization is less efficient

%% Evaluation of the CS Recovery Probability
q = 1000;
slist = 14:2:42;
Slist = slist(mod(0:q-1,length(slist))+1);

%%per
% Genetate signals so that |x0(:,j)| has sparsity |Slist(i)|.

U = rand(n,q);
v = sort(U);
v = v( (0:q-1)*n + Slist );
x0 = U<=repmat( v, [n 1] );

y = A*x0;

%% Exo 3:
proba = []; tol = 10/n;
tx = x0;
niter = 800;
for i=1:niter
    tx = (1-mu/2)*tx + mu/2*rproxG( rproxF(tx,y),gamma );
end
x = proxF(tx,y);
E = mean(abs(x-x0))< tol;
for j=1:length(slist)
    proba(end+1) = mean(E(Slist==slist(j)));
end
figure(1); clf;
plot(slist, proba, 'color','k','linewidth',2);
xlabel('sparsity')
ylabel('CS recovery probability')
axis([14 42 -.05 1.05]);
