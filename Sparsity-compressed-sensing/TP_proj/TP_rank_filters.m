%% Settings:
clearvars;close all;clc;
set(0,'DefaultAxesFontSize',14)
set(0,'DefaultTextFontSize',14)

% Loading toolboxes:
addpath(genpath('matlab-toolboxes/toolbox_signal'))
addpath(genpath('matlab-toolboxes/toolbox_general'))

%% ----------------------------------
%% Rank Filters for Image Processing: 
%% ----------------------------------
%% Continuous Rank Filtering
%% Patches in Images
n = 256;
name = 'hibiscus';
f0 = load_image(name, n);
f0 = rescale(crop( sum(f0,3) ,n));
figure(1); clf;
imageplot(f0);
% noise level:
sigma = .04;
% Generate noisy image:
f = f0 + randn(n,n)*sigma;
figure(1); clf;
imageplot(clamp(f));

%% Patches extraction:
w = 3;
w1 = 2*w+1;

[Y,X] = meshgrid(1:n,1:n);
[dY,dX] = meshgrid(-w:w,-w:w);
dX = reshape(dX, [1 1 w1 w1]);
dY = reshape(dY, [1 1 w1 w1]);
X = repmat(X, [1 1 w1 w1]) + repmat(dX, [n n 1 1]);
Y = repmat(Y, [1 1 w1 w1]) + repmat(dY, [n n 1 1]);

X(X<1) = 2-X(X<1); Y(Y<1) = 2-Y(Y<1);
X(X>n) = 2*n-X(X>n); Y(Y>n) = 2*n-Y(Y>n);
% Patch extractor operator
Pi = @(f)reshape( f(X + (Y-1)*n), [n n w1*w1] );
P = Pi(f);
% Displaying some patches:
figure(1);clf;
for i=1:16
    x = floor( rand*(n-1)+1 );
    y = floor( rand*(n-1)+1 );
    imageplot( reshape(P(x,y,:,:), w1,w1), '', 4,4,i );
end

%% Linear Filter

Pmean = @(f)mean(Pi(f),3);
p = 100;
psi = @(f)f.^(1/p);
ipsi = @(f)f.^p;
figure(1);clf;
imageplot(Pmean(f),'filtered',1,2,1);
imageplot(Pmean(abs(f)) - ipsi(Pmean(psi(abs(f)))),'? contrast invariant',1,2,2);

%% Opening and Closing Rank Filters
%% Exo 1:
r = @(beta)min(ceil(beta*w1*w1)+1,w1*w1);
subsample = @(x,s)x(:,:,s);
phi = @(f,beta)subsample(sort(Pi(f), 3), r(beta));

betas = linspace(0,1,5);
figure(1); clf;
set(1,'units','normalized','position',[.1 .1 .7 .4])

for i=1:length(betas);
    beta = betas(i);
    imageplot(phi(f,beta), sprintf('\\beta= %.2f\nsnr = %.2f', beta, snr(f0,phi(f,beta))), 1,5,i);
end
%% closing (beta = 0)
closing = @(f)phi(f,0);
figure(1);clf;
imageplot(closing(f),'closing');

%% opening (beta = 1)
opening = @(f)phi(f,1);
figure(1); clf;
imageplot(opening(f),'opening');

%% Exo 2|3: (closing o opening)  and (opening o closing)
oc = @(f) opening(closing(f));
co = @(f) closing(opening(f));
figure(1); clf;
imageplot(oc(f),'opening\circclosing',1,2,1);
imageplot(co(f),'closing\circopening',1,2,2);

%% Exo 4: Iterative opening and closing

figure(1); clf;
set(1,'units','normalized','position',[.1 .1 .5 .7])

fo = f; fc = f; foc = f; fco = f;
for i = 1:4
    fo = opening(fo);
    fc = closing(fc);
    foc = oc(foc);
    fco = co(fco);
    imageplot(fo, sprintf('opening, it %d',i), 4,4, i);
    imageplot(fc, sprintf('closing, it %d',i), 4,4, i+4);
    imageplot(foc, sprintf('open\\circclos, it %d',i), 4,4, i+8);
    imageplot(fco, sprintf('clos\\circopen, it %d',i), 4,4, i+12);
end

%% Median Filter

medfilt = @(f)phi(f,1/2);
figure(1);clf;
imageplot(medfilt(f),'Median filter');

%% Exo 5 - iterative median filtering
f1 = f;
figure(1); clf;
for i=1:6
    f1 = medfilt(f1);
    imageplot(f1, sprintf('iteration %d',i), 2,3, i);
end
% Final result
figure(2); clf;
imageplot(f1, sprintf('Median , snr = %.3f',snr(f0,f1)));

%%
figure(1);clf;
list = round(linspace(1,256,10));
for i=1:length(list)
    subplot(2,5,i)
    imshow(reshape(P(list(i),list(i),:),[w1 w1]))
end