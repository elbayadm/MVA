function [data] = preprocess(varargin)
% params.k  K for k-fold cross validation
%
% params.do_trim trimming and enlarging the full set
%   --  params.sz : the new image size
%
% params.do_deslant Peform slant removal
%
% params.do_norm normalize the features of the full set
% params.hog : Add hog features
%   -- parmas.cs  : Cellsize for HOG 
%
% params.do_rotate Add virtual data by rotating samples
% params.angle : max rotation angle
% params.rot   : ratio of rotated images 
   
% Output:
    data=[];
    
    train = 0;
    params = varargin{1};
    X      = varargin{2};
    if nargin == 3 
        train = 1; % Training set
        Y  = varargin{3};
    end
    
    if params.do_trim | params.do_deslant | params.do_blur | params.do_median
	    % Improve the image quality
	    X = raw_processing(X,params);
    end
    
    
    if params.do_norm
	    % Features normalization
	    fprintf('Scaling the full set..')
	    	X = scale(X);
	    fprintf('done\n')
    end
    if ~train % Test set
        if params.hog
            if params.do_trim
                X = hog(X,params.cs, params.sz);
            else
                X = hog(X,params.cs, [28 28]);
            end
        end
        data = X;
    else      % Training set
    
        disp('Shuffling...')
        % Shuffle
%         I = randperm(length(Y));
%         X = X(I,:);
%         Y = Y(I);

        if params.do_rotate
            % Rotating the images
            [X,Y] = virtual_data( X,Y,params.rot,params.angle,params.sz);
        end

        if params.hog
            if params.do_trim
                X = hog(X,params.cs, params.sz);
            else
                X = hog(X,params.cs, [28 28]);
            end
        end
        k= params.k;
        if k>1
            disp('K-fold training and validation sets...')
            n = length(Y);
            Xtr  = {}; Ytr={};
            Xval = {}; Yval={};
            I = floor(linspace(1,n,k+1));
            for i=1:k
                v = I(i):I(i+1);
                t = setdiff((1:n)', v);
                Xval{end+1} = X(v,:);
                Xtr{end+1}  = X(t,:);
                Yval{end+1} = Y(v);
                Ytr{end+1}  = Y(t);
            end
        else
            Xtr = {X};
            Ytr = {Y};
            Yval = {};
            Xval = {};

        end
       data.Xtr  = Xtr;
       data.Xval = Xval;
       data.Ytr  = Ytr;
       data.Yval = Yval;
    end
   disp('Data processing completed')
end