Xfull = importdata('Xtr.csv');
%%
clc;
i = 44;
i = 56;
i = 401;

sigma=.8;
%th = 0.3;
sz = [28 28];
im0  = reshape(Xfull(i,:),sz);
% Detect edges:
BW = edges(im0);
BW = im0>.5; 
tic,
[T, R ,H] = houghTransform(im0);
H_ = hough(BW);
figure(2);clf;
subplot(1,5,1)
imshow(im0)
subplot(1,5,2)
imshow(BW)
subplot(1,5,3)
imshow(H)
subplot(1,5,4)
imshow(H_)
%%
clc; close all;
im0  = reshape(Xfull(123,:),sz);
BW = im0>.5; 

[T, R ,H] = houghTransform(BW);
[~,b]=max(max(H));
[~,a]=max(H(:,b));
fprintf('H(a,b)=%d\n',H(a,b))
fprintf('angle =%.2f\n',T(b))

data=var(H); 
[~, d] = max(data);
[~, c] = max(H(:,d));
fprintf('H(c,d)=%d\n',H(c,d))
fprintf('var angle =%.2f\n',T(d))   
% Line R(a) = x cos(T(b)) + y sin(T(b))
slope1 = [R(a), -cosd(T(b))]/sind(T(b));
slope2 = [R(c), -cosd(T(d))]/sind(T(d));
figure,
subplot(1,4,1)
imshow(im0)
hold on,
X=1:28;
Y1=slope1(1)+slope1(2)*(1:28);
Y2=slope2(1)+slope2(2)*(1:28);
plot(X,Y1,'r')
hold on,
plot(X,Y2,'b')

im1=imrotate(im0,T(b), 'bicubic');
subplot(1,4,2)
imshow(im1)

im2=imrotate(im0,T(d), 'bicubic');
subplot(1,4,3)
imshow(im2)

[im3,rot] = deslant_im(im0);
subplot(1,4,4)
imshow(im3)
title(int2str(rot))


%%
sz = [28 28];
im0  = reshape(Xfull(123,:),sz);
% Remove noise:
im1 = median_im (im0);
% Trim the image
im2 = trim(im1);
%im_=imgaussfilt(im_,sigma);
im2= imresize(im2,sz);
im3= imadjust(im2);

figure(2);clf;
subplot(1,5,1)
imshow(im0)
subplot(1,5,2)
imshow(im1)
subplot(1,5,3)
imshow(im2)
subplot(1,5,4)
imshow(im3)

%%
clc;
sz=[28 28];
figure,

im0  = reshape(Xfull(2,:),sz);
subplot(1,4,1)
imshow(im0)
im1=deslant_im(im0);
subplot(1,4,2)
imshow(im1)
% Remove noise with median filter:
%im2=median(im1);
th = .1;
im2=im1;
for r = 2:size(im2,1)-1
    for c = 2:size(im2,2)-1
        window = im1(r-1:r+1,c-1:c+1);
        window = window(:);
        window = sort(window, 'descend');
        im2(r,c)= window(5);
    end
end
im2=im1.*(im2>max(max(im2))/4);
subplot(1,4,3)
imshow(im2)
% Trim the image
im2 = trim(im2);
subplot(1,4,4)
imshow(im2)
im2= imresize(im2,sz);

