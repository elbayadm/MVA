function im_ = adjust(im)
adj = @(x) (x - min(x(:)))/(max(x(:)) - min(x(:)));
im_ = im; 
for i = 1:size(im,3) % adjust every channel
    im_(:,:,i) = adj(im(:,:,i));
end
