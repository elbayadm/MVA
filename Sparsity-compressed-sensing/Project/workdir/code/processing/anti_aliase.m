function im_ = anti_aliase(im,freq)
% anti aliasing with butterworth filter
F = butterworth(freq*3,freq*0.9);
mm = conv2(ones(size(im(:,:,1))),F,'same');
a1 = max(min(conv2(single(im(:,:,1)),F,'same'),1),0)./mm;
a2 = max(min(conv2(single(im(:,:,2)),F,'same'),1),0)./mm;
a3 = max(min(conv2(single(im(:,:,3)),F,'same'),1),0)./mm;
im_ = cat(3,a1,a2,a3);
end

function k = butterworth(width, f)
cf = 0.5 / f;
range = (-width:width)/(2*width);
[i,j] = ndgrid(range,range);
r = sqrt(i.^2+j.^2);
k = ifftshift(1./(1+(r./cf).^4));
k = fftshift(real(ifft2(k)));
k = k./sum(k(:));
end
