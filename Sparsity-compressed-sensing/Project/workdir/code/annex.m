clearvars; close all; clc;
set(0,'DefaultTextFontSize',14)
addpath(genpath('/usr/local/cellar/vlfeat'))
% Loading toolboxes:
addpath(genpath('matlab-toolboxes/toolbox_signal'))
addpath(genpath('matlab-toolboxes/toolbox_general'))
addpath(genpath('Project'))
col = [9 125 18;145 145 145];

%% Rgb to hsv:
n = 256*2;
name = 'hibiscus';
f0 = load_image(name, n);
f0 = adjust(f0);

tic,
hsv_m = rgb2hsv(f0);
V_m = adjust(hsv_m(:,:,3));
toc,
tic,
hsv = rgb2hsv1 (f0);
V = adjust(hsv(:,:,3));
toc
tic
hsv1 = rgb2hsv2 (f0);
V1 = hsv1(:,:,3);
toc
figure(1); clf
imageplot({V_m, V1},{'matlab','my version'},2,1)

rgb1 = hsv2rgb2(hsv1);
figure(2);clf
imageplot({f0 rgb1},{'original','recovered'},1,2)

%% Histograms: 
sx = 2;
width = sx;
sv = .1;  samp= 2; m = floor(samp/sv);

[s_list,fs, Ds, Rs] = smoothed_histograms(V1,width,sx,sv,samp,{'hist','derivative','integral'});
y= 123; x = 104;
fp = squeeze(fs(y,x,:));
Rp = squeeze(Rs(y,x,:));

figure(1);clf;
set(1,'units','normalized','position',[.1 .1 .4 .3])

subplot(1,2,1)
plot(s_list,fp,'color','b','linewidth',2)
hold on,
gridlines(s_list,1,[.2,.2,.2,.7],':')
xlabel('intensity')
title('frequency')
set(gca,'ytick',[])

subplot(1,2,2)
plot(s_list,Rp,'color','b','linewidth',2)
hold on,
gridlines(s_list,1,[.2,.2,.2,.7],':')
hold on,
gridlines([.05 .5 .95],1,'r','--',2)
xlabel('intensity')
title('Integral')
axis 'tight'
axis 'square'
set(gca,'ytick',[.05 .5 .95])

%% Spatial kernel
sx = 3;
width = 4;

[gridX, gridY] = meshgrid( 0 : (2*width), 0 : (2*width));
gridX = gridX - width;
gridY = gridY - width;
gridRSquared = ( gridX .* gridX + gridY .* gridY);
GaussianKernel = @(s) exp(gridRSquared/2/s^2);
W = GaussianKernel(sx); W = W/sum(W(:));
square = ones(size(W));
square = square/sum(square(:));
figure(1);clf
set(1,'units','normalized','position',[.1 .1 .5 .3])
subplot(1,2,1)
imagesc(W)
text(width+1,width+1,'\bullet', 'HorizontalAlignment','center',...
'VerticalAlignment','middle',...    
'color','w','fontsize',18,'fontweight','bold');
hold on,
gridlines(0.5:(2*width+1.5),1,'k','-')
hold on,
gridlines(0.5:(2*width+1.5),1,'k','-',2)
set(gca,'xtick',[],'ytick',[])
title('Octagonal kernel')
subplot(1,2,2)
imagesc(square)
text(width+1,width+1,'\bullet', 'HorizontalAlignment','center',...
'VerticalAlignment','middle',...    
'color','w','fontsize',18,'fontweight','bold');
hold on,
gridlines(0.5:(2*width+1.5),1,'k','-')
hold on,
gridlines(0.5:(2*width+1.5),1,'k','-',2)
title('Unit square kernel')
set(gca,'xtick',[],'ytick',[])
colormap gray
