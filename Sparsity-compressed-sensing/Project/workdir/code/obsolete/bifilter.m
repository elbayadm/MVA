function output  = bifilter(im, sx, sv, p ,method)
% Bilateral filter by stacking 
% http://www.numerical-tours.com/matlab/denoisingadv_8_bilateral/

% im with a single channel
assert(size(im,3)==1)

[h, w] = size(im);
im  = imresize(im, 2*[h w]);

% map values to [0 1]
m  =  min(im(:));
M  =  max(im(:));
im = (im - m )/( M - m);

% Gaussian filter:
x = [0:w-1, -w:-1];
y = [0:h-1, -h:-1];
[Y,X] = meshgrid(x,y);
GaussianFilt = @(s)exp((-X.^2-Y.^2)/(2*s^2));

% Gaussian filtering over the fourier domain
Filter = @(F,s)real( ifft2( fft2(F).*repmat( fft2(GaussianFilt(s)), [1 1 size(F,3)] ) ) );
Gaussian = @(x,sigma)exp( -x.^2 / (2*sigma^2) );

% Weights
W = Gaussian( repmat(im, [1 1 p]) - ...
    repmat( reshape((0:p-1)/(p-1), [1 1 p]) , [2*h 2*w 1]), sv );

% Stacks
F = Filter(W.*repmat(im, [1 1 p]), sx) ./ Filter(W, sx);
% Destacking
[y,x] = meshgrid(1:(2*h),1:(2*w));
destacking = @(F,I) F(x + 2*(y-1)*h + (I-1)*4*h*w);

switch method
    case 'NN'
        % nearest neighbor
        output = destacking(F,round(im*(p-1))+ 1);
    case 'interp1'
    % 1st order interpolation
    frac = @(x)x-floor(x);
    lininterp = @(f1,f2,Fr)f1.*(1-Fr) + f2.*Fr;
    output = lininterp( destacking(F, clamp(floor(im*(p-1)) + 1,1,p) ), ...
                                  destacking(F, clamp(ceil(im*(p-1)) + 1,1,p) ), ...
                                  frac(im*(p-1)) );
    otherwise
        fprintf(2,'unknown method');
end

output = imresize(output , [h w]);

