function X_ = deslant(X)
global doviz
X_= X;
 textprogressbar('Removing the images slant:  ');
 n=size(X,1);
    for i=1:n
        textprogressbar(i,n);
        im  = reshape(X(i,:),[28 28]); 
        im_ = im_deslant(im);
        if (rand(1)>.95) && doviz
            figure(1);clf;
            vl_tightsubplot(2,1,'box','inner') ;
            imshow(im) ;
            text(3,3,'Original',...
           'background','w') ;
            vl_tightsubplot(2,2,'box','inner') ;
            imshow(im_) ;
            text(3,3,'Deslanted',...
           'background','w') ;
            drawnow
        end
        X_(i,:) = im_(:);
    end
    textprogressbar('done');
end

 