function im_ = im_resize(im,sz)
% resize an image using bicubic interpolation
r_ = sz(1);
c_ = sz(2);
d = size(im,3);

if d==3
    % RGB image
    im_ = zeros(r_,c_, size(im,3));
    for m=1:size(im,3)
        im_(:,:,m) = im_resize(im(:,:,m), [r_ c_]);
    end
    return;
end
r = size(im,1);
c = size(im,2);
[Y,X] = meshgrid( (0:c-1)/(c-1), (0:r-1)/(r-1) );
[YI,XI] = meshgrid( (0:c_-1)/(c_-1), (0:r_-1)/(r_-1) );
im_ = interp2( Y,X, im, YI,XI ,'cubic');