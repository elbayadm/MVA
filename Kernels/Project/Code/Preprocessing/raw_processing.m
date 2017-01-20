function X_ = raw_processing(X,params)
global doviz
sz = params.sz;
% Gaussian blur width:
sigma = 1;
X_=zeros(size(X,1),sz(1)*sz(2));
 textprogressbar('Processing the raw images:  ');
 n=size(X,1);
    for i=1:n
        textprogressbar(i,n);
        im  = reshape(X(i,:),[28 28]);
        im1 = im;
        
        if params.do_deslant
        % Remove the slant
        im1 = im_deslant(im1);
        end
        
        if params.do_trim
        % Trim the image
        im1 = im_trim(im1);
        end
        
        if params.do_blur
        % Remove noise with gaussian filter:
        im1 = im_gaussianblur(im1,sigma);
        end
        
        if params.do_median
        im1 = im_rankfilter(im1,2,.5);
        end
        
        im1 = im_resize(im1,sz);
        im1 = im_adjust(im1);
        if (rand(1)>.95) && doviz
            figure(1);clf;
            vl_tightsubplot(2,1,'box','inner') ;
            imshow(im) ;
            text(3,3,'Original',...
           'background','w') ;
            vl_tightsubplot(2,2,'box','inner') ;
            imshow(im1) ;
            text(3,3,'Processed',...
           'background','w') ;
            drawnow
        end
        X_(i,:) = im1(:);
    end
    textprogressbar('done');
end

 