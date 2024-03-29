function im_ = im_gaussianblur(im, sigma)
width = sigma;
[gridX, gridY] = meshgrid( 0 : (2*width), 0 : (2*width));
gridX = gridX - width;
gridY = gridY - width;
gridRSquared = ( gridX .* gridX + gridY .* gridY);
KK = exp(-gridRSquared/2/sigma^2);
KK = KK/sum(KK(:));
im_ = convn(im, KK,'same');
