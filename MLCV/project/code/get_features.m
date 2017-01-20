function [fts,crds,idxs] =  get_features(input_image,feat,points,cnn_net)
do_dense = (nargin==2) || isempty(points);

crds = [];
idxs = [];
%% some pre-processing
inim = input_image;
if size(input_image,3)>1 
    inim = single(rgb2gray(inim)); 
else
    inim = single(inim);
end
inim = inim./max(inim(:));
feat = lower(feat);
switch feat,
    case 'sift',
        binSize = 4;
        %% magical number from the documentation of vl_dsift
        magnif  = 3;
        if do_dense
            %% type>>  help vl_dsift  for more justifications
            %%
            Is = vl_imsmooth(inim, sqrt((binSize/magnif)^2 - .25)) ;
            [crds, fts] = vl_dsift(Is, 'size', binSize);
            f(3,:) = binSize/magnif ;
            f(4,:) = 0;
        else
            magnif      = 3;
            f           = [points;[binSize/magnif;0]*ones(1,size(points,2))];
            [crds,fts ] = vl_sift(inim, 'frames', f) ;
        end
        fts = double(fts);
        fts = fts./repmat(max(sqrt(sum(fts.*fts,1)),.01),[128,1]);
    case 'steer',
        filterbank     = construct_filterbank(.8);
        filter_dense   = apply_filterbank(inim,filterbank);
        
        [nfeat,sv,sh]  = size(filter_dense);
        
        if do_dense
            fts            = reshape(filter_dense,[nfeat,sv*sh]);
            [cr_h,cr_v]    = meshgrid([1:sh],[1:sv]);
            crds           = [cr_h(:)';cr_v(:)'];
        else
            npoints = size(points,2);
            fts = zeros(nfeat,npoints);
            for k=1:npoints
                fts(:,k) = filter_dense(:,points(2,k),points(1,k));
            end
        end
    case {'cnn2', 'cnn5'}
        assert(nargin>=4, 'You must provide a pre-trained CNN model');
        if ismatrix(input_image), input_image = repmat(input_image, [1 1 3]); end
        if max(input_image(:)) <= 1 && min(input_image(:)) >=0
            input_image = input_image * 255;
        end
        % Image has to be normalized first (resize the ImageNet mean image 
        % to the same size as the input image and subtract it).
        imsize = size(input_image);
        %fprintf('size(image): %dx%dx%d - ',imsize)        
        mean_image = imresize(cnn_net.meta.normalization.averageImage, imsize(1:2), 'bilinear');
        %fprintf('size(mean image): %dx%d - ',size(mean_image))
        input_image_normalized = single(input_image) - mean_image;
        % Compute the network responses    
        if strcmp(feat,'cnn2')
            layer = 7;
            res = init_res(layer) ; 
            res(1).x = input_image_normalized;
            res = forwardto_nn(cnn_net, layer, res) ;
            % res = vl_simplenn(cnn_net, input_image_normalized);
            fts = res(layer).x ;
        elseif strcmp(feat, 'cnn5')
            layer = 15;
            res = init_res(layer) ; 
            res(1).x = input_image_normalized;
            res = forwardto_nn(cnn_net, layer, res) ;
            % res = vl_simplenn(cnn_net, input_image_normalized);
            fts = res(layer).x ;
        end
        fts = imresize(fts, imsize(1:2), 'nearest');
        fts = reshape(fts, [],size(fts,3))'; 
        normfact = sqrt(sum(fts.^2,1)); normfact(normfact == 0) = 1;
        fts = bsxfun(@rdivide,fts, normfact); % normalize
        if do_dense
            [cr_h,cr_v] = meshgrid([1:imsize(2)],[1:imsize(1)]);
            crds= [cr_h(:)';cr_v(:)'];
        else
            crds= [points(1,:); points(2,:)];
            idxs= imsize(1)*(crds(1,:)-1) + crds(2,:);
            fts = fts(:,idxs);
        end
end

fts(end+1,:) = 1; %% append the DC term at the end

if do_dense
    [sv,sh] = size(input_image);
    coord_v = crds(2,:);
    coord_h = crds(1,:);
    
    %% matlab indexing
    idxs    = sv*(coord_h-1) + coord_v;
end
