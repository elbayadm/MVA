function output = bilateral_filter(im,sx,sv)
% Implementation of:
% Paris and Durand (2006):
% A Fast Approximation of the Bilateral Filter 
% using a Signal Processing Approach 

% Parameters:
% sx : spatial width
% sv : value width

[h,w,~] = size(im);

m = min(im(:));  M = max(im(:)); delta = M - m;
 
paddingXY = 3;
paddingZ = 3;

% 3D grid:
downsampledWidth = floor( ( w - 1 ) / sx ) + 1 + 2 * paddingXY;
downsampledHeight = floor( ( h - 1 ) / sx ) + 1 + 2 * paddingXY;
downsampledDepth = floor( delta / sv ) + 1 + 2 * paddingZ;

gridIm = zeros( downsampledHeight, downsampledWidth, downsampledDepth );
gridWeights = zeros( downsampledHeight, downsampledWidth, downsampledDepth );

% compute downsampled indices
[ jj, ii ] = meshgrid( 0 : w - 1, 0 : h - 1 );
di = round( ii / sx ) + paddingXY + 1;
dj = round( jj / sx ) + paddingXY + 1;
dz = round( ( im - m ) / sv ) + paddingZ + 1;
% Scattering
for k = 1 : numel( dz )
    imZ = im( k ); 
    dik = di( k );
    djk = dj( k );
    dzk = dz( k );

    gridIm( dik, djk, dzk ) = gridIm( dik, djk, dzk ) + imZ;
    gridWeights( dik, djk, dzk ) = gridWeights( dik, djk, dzk ) + 1;
end


% Gaussian kernel
[gridX, gridY, gridZ] = meshgrid( 0 : 2, 0 : 2, 0 : 2 );
gridX = gridX - 1;
gridY = gridY - 1;
gridZ = gridZ - 1;
gridRSquared = ( gridX .* gridX + gridY .* gridY ) ./ (1 + gridZ .* gridZ );
kernel = exp( -0.5 * gridRSquared );

% convolve
blurredGridIm = convn( gridIm, kernel, 'same' );
blurredGridWeights = convn( gridWeights, kernel, 'same' );


% divide
blurredGridWeights( blurredGridWeights == 0 ) = -2; % avoid divide by 0, won't read there anyway
normalizedBlurredGrid = blurredGridIm ./ blurredGridWeights;
normalizedBlurredGrid( blurredGridWeights < -1 ) = 0; % put 0s where it's undefined

% upsample
[ jj, ii ] = meshgrid( 0 : w - 1, 0 : h - 1 ); % meshgrid does x, then y, so output arguments need to be reversed
% no rounding
di = ( ii / sx ) + paddingXY + 1;
dj = ( jj / sx ) + paddingXY + 1;
dz = ( im - m ) / sv + paddingZ + 1;

% interpn takes rows, then cols, etc
% i.e. size(v,1), then size(v,2), ...
output = interpn( normalizedBlurredGrid, di, dj, dz );

