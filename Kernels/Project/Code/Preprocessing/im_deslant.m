function [im_, varargout] = im_deslant(im)
    lower_tol = -70;
    upper_tol = 40;
    rot = 0;
    angle = slant_detection(im);
    if angle>lower_tol && angle < upper_tol
        rot = 1;
        im_= imrotate(im,angle, 'bicubic');
        im_= im_resize(im_,size(im));
    else
        im_=im;
    end
    if nargout > 1
        varargout{1} = rot;
        if nargout > 2
            varargout{2} = angle;
        end
    end

    
end
    


