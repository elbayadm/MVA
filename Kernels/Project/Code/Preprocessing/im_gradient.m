function varargout = im_gradient(im,polar)
    % Image gradient in polar/cartesian coordinates
    Gx = imfilter(im,[1 0; 0 -1],'replicate');         
    Gy = imfilter(im,[0 1; -1 0],'replicate'); 
    if polar
        % Tranforming the gradient vectors to polar form
        Gdir = atan2(Gy,Gx)*180/pi;
        Gmag = hypot(Gx,Gy);
        varargout{1} = Gdir;
        varargout{2} = Gmag; 
    else
        varargout{1} = Gx;
        varargout{2} = Gy; 
    end

end