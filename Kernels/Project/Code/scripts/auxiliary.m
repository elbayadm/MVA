% Visualize samples and features
set(0,'DefaultAxesFontSize',12)
set(0,'DefaultTextFontSize',12)
%% Features
figure(1);clf;
M = min(Xfull,[],1);
plot(M)
%% Visualize HOG
im = reshape(Xfull(450,:),[28 28]);
H = im_hog(im, 7, 2, 12);
[matH, vis] = extractHOGFeatures(im,'CellSize',[7 7],'BlockOverlap',[1 1]);

imt = im_trim(im);
imt= im_resize(imt,[28 28]);
Ht = im_hog(imt, 7, 2, 12);
[matHt, vist] = extractHOGFeatures(imt,'CellSize',[7 7],'BlockOverlap',[1 1]);

figure(1);clf;
subplot(2,1,1)
plot(1:length(H),H,'-xb')
hold on,
plot(1:length(matH),matH,'-xr')

subplot(2,1,2)
plot(1:length(Ht),Ht,'-xb')
hold on,
plot(1:length(matHt),matHt,'-xr')

%% Deslanting angles:
%clc;close all;
figure(1);
for i=1:400
    im = reshape(Xfull(i,:),[28 28]);
    [im1, rot, angle] =  im_deslant(im);
    if rot & (Yfull(i)==7)
        fprintf('%d : %.2f\n',i,angle);
        subplot(1,2,1)
            imshow(im)
        subplot(1,2,2)
            imshow(im1)
        drawnow,
    end
end
%%  Pre-processing ilustration
i = 2817; %6
%i = 401;  %!
i = 1840;
f0 = reshape(Xfull(i,:),[28 28]);
[t , r] = slant_detection(f0);
f1 =  im_deslant(f0);
f2 = im_trim(f1);
f2 = im_resize(f2,[28 28]);
f3 = im_rankfilter(f2,2,.5);
f4 = im_gaussianblur(f2,2);
figure(2); clf;
vl_tightsubplot(1,5,1,'marginright',.01,'marginleft',.01)
imshow(f0)
hold on,
plot(r-xlim*sind(t),xlim,'color','r','linewidth',3)
title('Skewed')

vl_tightsubplot(1,5,2,'marginright',.01,'marginleft',.01)
imshow(f1)
title('Deskewed')

vl_tightsubplot(1,5,3,'marginright',.01,'marginleft',.01)
imshow(f2)
title('Trimmed')

vl_tightsubplot(1,5,4,'marginright',.01,'marginleft',.01)
imshow(f3)
title('Median')

vl_tightsubplot(1,5,5,'marginright',.01,'marginleft',.01)
imshow(f4)
title('Blurred (\sigma=2)')

%print '-f2' 'figures/prep' '-dpdf'


%% Visualize samples:
figure(1)
m = size(Xfull,1);
I = randi(m,20);
for i=1:20
    vl_tightsubplot(20,i)
    imshow(reshape(Xfull(I(i),:),[28 28]))
    text(2,2,sprintf('%d',I(i)),'background','w')
    drawnow
end