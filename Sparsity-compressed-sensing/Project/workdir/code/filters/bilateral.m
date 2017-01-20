function [output, fs] = bilateral(I,width,sx,sv,T)
% Bilateral filter
% smoothed, locally-weighted histogram
% Gaussian kernel:
tic,
[h w] = size(I);
[gridX, gridY] = meshgrid( 0 : (2*width), 0 : (2*width));
gridX = gridX - width;
gridY = gridY - width;
gridRSquared = ( gridX .* gridX + gridY .* gridY);
GaussianKernel = @(s) exp(gridRSquared/2/s^2);

% Spatial kernel
W = GaussianKernel(sx); W = W/sum(W(:));

% shifts:
m = floor(T/sv);
s_list = linspace(0,1,m);
II = bsxfun(@minus, repmat(I,[1 1 m]), reshape(s_list,[1,1,m]));
K = exp( -II.^2 / (2*sv^2) );
fs = convn(K.*repmat(I,[1 1 m]),W,'same');
ws = convn(K,W,'same');
fs = fs./ws;
step = s_list(2)-s_list(1);
J = 1+floor(I/step);
s_previous = (J-1)*step;
f_previous = map_matrix(fs,J);
f_next = map_matrix(fs,J+1);
output = s_previous + (f_next - f_previous)/step .* (I - s_previous);
fprintf('Bilateral computation : %.2fs\n',toc)
