%% Settings:
clearvars;close all;clc;
set(0,'DefaultAxesFontSize',14)
set(0,'DefaultTextFontSize',14)

% Loading toolboxes:
addpath(genpath('matlab-toolboxes/toolbox_signal'))
addpath(genpath('matlab-toolboxes/toolbox_general'))

%% -------------------
%% Bilateral Filtering: 
%% -------------------
%% Gaussian Linear Filtering ( tends to blur edges )
% Loading the image
n = 256*2;
name = 'hibiscus';
f0 = load_image(name, n);
f0 = rescale(crop( sum(f0,3) ,n));

x = [0:n/2-1, -n/2:-1];
[Y,X] = meshgrid(x,x);
GaussianFilt = @(s)exp( (-X.^2-Y.^2)/(2*s^2) );
figure(1);clf;
imageplot(fftshift(GaussianFilt(40)));

%% linear Gaussian filtering over the Fourier domain
% per channel filtering:
Filter = @(F,s)real( ifft2( fft2(F).*repmat( fft2(GaussianFilt(s)), [1 1 size(F,3)] ) ) );

% Example:
figure(1);clf;
imageplot( Filter(f0,5) );

%% Exo 1:
figure(1);clf;
sigmas = [.5 3 6 10];
for i = 1:length(sigmas)
    s = sigmas(i);
    imageplot( Filter(f0,s),sprintf('\\sigma=%.2f',s),2,2,i);
    
end
%% Bilateral Filter
%% Bilateral Filter by Stacking
sx = 5;
sv = .2;
p = 10;
Gaussian = @(x,sigma)exp( -x.^2 / (2*sigma^2) );
WeightMap = @(f0,sv)Gaussian( repmat(f0, [1 1 p]) - repmat( reshape((0:p-1)/(p-1), [1 1 p]) , [n n 1]), sv );
W = WeightMap(f0,sv);

%% Exo 2: Displaying weights
figure(1);clf;
vlist = round([.1 .3 .6 .9]*p);
for i=1:4
    v = vlist(i);
    imageplot( W(:,:,v), ['v_i=' num2str((v-1)/(p-1),2)], 2,2,i );
end

% Shortcut to compute the bilateral stack
bileteral_stack_tmp = @(f0,sx,W)Filter(W.*repmat(f0, [1 1 p]), sx) ./ Filter(W, sx);
bileteral_stack = @(f0,sx,sv)bileteral_stack_tmp(f0,sx,WeightMap(f0,sv));
F = bileteral_stack(f0,sx,sv);

%% Exo 3 : displaying stacks
figure(1);clf;
vlist = round([.1 .3 .6 .9]*p);
for i=1:4
    v = vlist(i);
    imageplot( F(:,:,v), ['v_i=' num2str((v-1)/(p-1),2)], 2,2,i );
end
%% Destacking
[y,x] = meshgrid(1:n,1:n);
indexing = @(F,I)F(I);
destacking = @(F,I)indexing(F,x + (y-1)*n + (I-1)*n^2);
bilateral_nn = @(f0,sx,sv)destacking( bileteral_stack(f0,sx,sv), round( f0*(p-1) ) + 1 );

fNN = bilateral_nn(f0,sx,sv);
figure(1);clf;
imageplot( fNN );

%% Interpolation reconstruction
frac = @(x)x-floor(x);
lininterp = @(f1,f2,Fr)f1.*(1-Fr) + f2.*Fr;
bilateral_lin1 = @(F,f0)lininterp( destacking(F, clamp(floor(f0*(p-1)) + 1,1,p) ), ...
                                  destacking(F, clamp(ceil(f0*(p-1)) + 1,1,p) ), ...
                                  frac(f0*(p-1)) );
bilateral_lin = @(f0,sx,sv)bilateral_lin1(bileteral_stack(f0,sx,sv), f0);

%% Exo 4:
c = [.5 .4]*n; q = 200;
figure(1);clf;
imageplot( crop(bilateral_nn(f0,sx,sv),q,c) ,'NN',1,2,1);
imageplot( crop(bilateral_lin(f0,sx,sv),q,c) ,'Linear',1,2,2);

%% Exo 5: influence of sigma_x
figure(1); clf;
list_sx = [1 6 11 16];
for i= 1:length(list_sx)
    sx = list_sx(i);
    imageplot( bilateral_lin(f0,sx,.2), ['\sigma_x=' num2str(sx,2)], 2,2,i );
end

%% Exo 5: influence of sigma_v
figure(1); clf;
list_sv = [5 17 28 40]/100;
for i= 1:length(list_sv)
    sv = list_sv(i);
    imageplot( bilateral_lin(f0,8,sv), ['\sigma_v=' num2str(sv,2)], 2,2,i );
end

