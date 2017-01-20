%% Settings:
clearvars;close all;clc;
set(0,'DefaultAxesFontSize',14)
set(0,'DefaultTextFontSize',14)
% Loading toolboxes:
addpath(genpath('matlab-toolboxes/toolbox_signal'))
addpath(genpath('matlab-toolboxes/toolbox_general'))
addpath(genpath('workdir'))
col = [9 125 18;145 145 145];

%% Bilateral filtering:
f0 = im2double(imread('cormorant.jpg'));
f0 = imresize(f0, [380 300]);
f0 = adjust(f0);

hsv1 = rgb2hsv2 (f0);
V1 = hsv1(:,:,3);
V1  = adjust(V1);

sx = 3;
width = sx;
sv = .2;  samp= 5;
output = bilateral(V1,width,sx,sv,samp);
fb = hsv2rgb2(cat(3,hsv1(:,:,1),hsv1(:,:,2),output));
%fb = anti_aliase(fb,2);
figure(1); clf;
imageplot(fb)

%% Morphological filters:
clc;
n = 256;
name = 'hibiscus';
f0 = rescale( load_image(name,n) );
f0 = adjust(f0);
hsv1 = rgb2hsv2 (f0);
V1 = hsv1(:,:,3);
V1  = adjust(V1);
sx = 2;
width = sx;
sv = .2;  samp= 5;
beta = .5;
erosion = morphology(V1,width,sx,sv,samp,.05);
dilation = morphology(V1,width,sx,sv,samp,.95);
opening = morphology(erosion,width,sx,sv,samp,.95);
closing = morphology(dilation,width,sx,sv,samp,.05);

% closing followed by opening
morpho = morphology(closing,width,sx,sv,samp,.05);
morpho = morphology(morpho,width,sx,sv,samp,.95);

erosion = hsv2rgb2(cat(3,hsv1(:,:,1),hsv1(:,:,2),erosion));
dilation = hsv2rgb2(cat(3,hsv1(:,:,1),hsv1(:,:,2),dilation));
opening = hsv2rgb2(cat(3,hsv1(:,:,1),hsv1(:,:,2),opening));
closing = hsv2rgb2(cat(3,hsv1(:,:,1),hsv1(:,:,2),closing));
morpho = hsv2rgb2(cat(3,hsv1(:,:,1),hsv1(:,:,2),morpho));

m = floor(samp/sv);
figure(1);clf;
set(1,'units','normalized','position',[.1 .1 .6 .75]);
set(1,'PaperType','A4','PaperPositionMode','auto');

vl_tightsubplot(6,1,...
    'marginright',.01,'marginright',.01);
imageplot(f0);
title('Original image','fontsize',10);

vl_tightsubplot(6,2,...
    'marginright',.01,'marginright',.01);
imageplot(erosion);
title(sprintf('(Erosion) \\sigma_x = %d | \\sigma_v = %.2f | m = %d'...
    ,sx,sv,m),'fontsize',10);

vl_tightsubplot(6,3,...
    'marginright',.01,'marginright',.01);
imageplot(dilation)
title(sprintf('(Dilation) \\sigma_x = %d | \\sigma_v = %.2f | m = %d'...
    ,sx,sv,m),'fontsize',10);

vl_tightsubplot(6,4,...
    'marginright',.01,'marginright',.01);
imageplot(opening);
title(sprintf('(Opening) \\sigma_x = %d | \\sigma_v = %.2f | m = %d'...
   ,sx,sv,m),'fontsize',10);

vl_tightsubplot(6,5,...
    'marginright',.01,'marginright',.01);
imageplot(closing);
title(sprintf('(Closing) \\sigma_x = %d | \\sigma_v = %.2f | m = %d'...
    ,sx,sv,m),'fontsize',10);

vl_tightsubplot(6,6,...
    'marginright',.01,'marginright',.01);
imageplot(morpho);
title(sprintf('(Bousseau & al.) \\sigma_x = %d | \\sigma_v = %.2f | m = %d'...
   ,sx,sv,m),'fontsize',10);


%% Mode filters:
f0 = im2double(imread('tractor.jpg'));
f0 = imresize(f0, [340 500]);
f0 = adjust(f0);

hsv1 = rgb2hsv2 (f0);
V1 = hsv1(:,:,3);
V1  = adjust(V1);

sx = 3;
width = sx;
sv = .2;  samp= 5;
output = closest_mode(V1,width,sx,sv,samp);
fc = hsv2rgb2(cat(3,hsv1(:,:,1),hsv1(:,:,2),output));
%fc = anti_aliase(fc,2);
figure(1); clf;
imageplot(fc)

%% Selective diffusion:
clc;
f0 = im2double(imread('dog.jpg'));
f0 = imresize(f0, [400 500]);
f0 = adjust(f0);
I = 1:size(f0,1); J=1:size(f0,2);
hsv1 = rgb2hsv2 (f0);
V1 = hsv1(:,:,3);
V1  = adjust(V1);

sx = 2;
width = sx;
sv = .08;  samp= .5;

[DC, FC] = selective_diffusion(V1,width,sx,sv,samp,'bilateral');
ffc = hsv2rgb2(cat(3,hsv1(:,:,1),hsv1(:,:,2),FC));
fc = hsv2rgb2(cat(3,hsv1(:,:,1),hsv1(:,:,2),DC));
fce = hsv2rgb2(cat(3,hsv1(:,:,1),hsv1(:,:,2),DC+3*(V1-DC)));
fce2 = hsv2rgb2(cat(3,hsv1(:,:,1),hsv1(:,:,2),FC+3*(V1-FC)));

m = floor(samp/sv);
figure(1);clf;
vl_tightsubplot(2,3,1,...
    'marginright',.01,'marginright',.01);
imageplot(f0(I,J,:))
title('Original','fontsize',10)


vl_tightsubplot(2,3,2,...
    'marginright',.01,'marginright',.01);
imageplot(ffc(I,J,:))
title('Bilateral','fontsize',10)

vl_tightsubplot(2,3,3,...
    'marginright',.01,'marginright',.01);
imageplot(fc(I,J,:))
title('Base layer','fontsize',10)

vl_tightsubplot(2,3,5,...
    'marginright',.01,'marginright',.01);
imageplot(fce2(I,J,:)-fce(I,J,:))
title('+','fontsize',10)

vl_tightsubplot(2,3,4,...
    'marginright',.01,'marginright',.01);
imageplot(fce(I,J,:))
title('diffusion enhanced x2','fontsize',10)

vl_tightsubplot(2,3,6,...
    'marginright',.01,'marginright',.01);
imageplot(fce2(I,J,:))
title('Enhanced x2','fontsize',10)

%% Multi layer decomposition:
clc;
f0 = im2double(imread('moon.jpg'));
f0 = imresize(f0, [308 386]);
f0 = adjust(f0);
f0  = sum(f0,3)/3;

figure(1);clf;
vl_tightsubplot(2,3,1,...
    'marginright',.01,'marginright',.01);
imageplot(f0)
title('Original','fontsize',10)
drawnow

V1  = adjust(f0);
sx = 8;
width = sx; 
sv = .1;  samp= 1; m = floor(samp/sv);

[B1, ~] = selective_diffusion(V1,width,sx,sv,samp,'closest');
vl_tightsubplot(2,3,2,...
    'marginright',.01,'marginright',.01);
imageplot(B1)
title('B1','fontsize',10)
drawnow


D1 = adjust(V1 - B1);
[B2, ~] = selective_diffusion(D1,width,sx,sv/10,samp/4,'closest');

vl_tightsubplot(2,3,3,...
    'marginright',.01,'marginright',.01);
imageplot(B2)
title('B2','fontsize',10)
drawnow


D2 = adjust(D1 - B2);
[B3, ~] = selective_diffusion(D2,width,sx,sv/12,samp/5,'closest');
vl_tightsubplot(2,3,4,...
    'marginright',.01,'marginright',.01);
imageplot(B3)
title('B3','fontsize',10)
drawnow

D3 = adjust(D2 - B3);
vl_tightsubplot(2,3,5,...
    'marginright',.01,'marginright',.01);
imageplot(D3)
title('D3','fontsize',10)
drawnow

Enhanced = B1+3*B2+2*B3+D3;
vl_tightsubplot(2,3,6,...
    'marginright',.01,'marginright',.01);
imageplot(Enhanced)
title('Enhanced','fontsize',10)
drawnow
