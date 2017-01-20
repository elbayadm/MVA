function phi = im_rankfilter(im,width, beta)
% Beta : the rank (50% for the median, 0% for erosion and 100% for dilation)
% width : the window size

w1 = 2*width+1;
r = min(ceil(beta*w1*w1)+1,w1*w1);
P = extract_patches (im, width);
P = sort(P, 3);
phi = P(:,:, r);
end

function P = extract_patches (im,width)
% Extract patches of input image im
% Each patch of size (2*width+1)*(2*width+1)
[h,w]  = size(im);
w1 = 2*width+1;
[Y,X] = meshgrid(1:w,1:h);
[dY,dX] = meshgrid(-width:width,-width:width);
dX = reshape(dX, [1 1 w1 w1]);
dY = reshape(dY, [1 1 w1 w1]);
X = repmat(X, [1 1 w1 w1]) + repmat(dX, [h w 1 1]);
Y = repmat(Y, [1 1 w1 w1]) + repmat(dY, [h w 1 1]);

X(X<1) = 2-X(X<1); Y(Y<1) = 2-Y(Y<1);
X(X>h) = 2*h-X(X>h); Y(Y>w) = 2*w-Y(Y>w);
P = reshape( im(X + (Y-1)*h), [h w w1*w1] );
end