function X_ = hog(X, cs, sz)
    % -------------
        blocksize = 2;
        nbins     = 8; 
    % --------------
    im  = reshape(X(1,:),sz); 
    H = im_hog(im, cs, blocksize, nbins);
    hh = length(H);
    [n,d]=size(X);
    X_=zeros(n,d+hh);
    textprogressbar(sprintf('Computing HOG features (dim = %d):  ',hh));

    for i=1:n
        textprogressbar(i,n);
        im  = reshape(X(i,:),sz);
        H = im_hog(im, cs, blocksize, nbins);
        X_(i,:)=[X(i,:) H'];
    end
    textprogressbar('done');
    fprintf('\nNew Training set X :(%d x %d)\n',size(X_));
end

