function [net, info] = cnn_finetune(varargin)
%CNN_IMAGENET   Demonstrates training a CNN on ImageNet
%  This demo demonstrates training the AlexNet, VGG-F, VGG-S, VGG-M,
%  VGG-VD-16, and VGG-VD-19 architectures on ImageNet data.

opts.dataDir = '../data' ;
opts.expDir = fullfile(opts.dataDir, 'finetune') ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = false ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.numFetchThreads = 12 ;
opts.lite = false ;

% start with a learning rate of 0.01 and gradually decrease it
opts.train.learningRate = logspace(-2,-5, 20); 
opts.train.weightDecay = 0.0005;
opts.train.momentum = 0.9;
opts.train.batchSize = 5;
opts.train.plotDiagnostics = false;
opts.train.plotStatistics  = false;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

net = cnn_init() ;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------
imdb = setupImdb(opts,net.meta,'train');
imdb.images.data = bsxfun(@minus, imdb.images.data, net.meta.normalization.averageImage);
% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainFn = @cnn_train ;
  case 'dagnn', trainFn = @cnn_train_dag ;
end
[net, info] = trainFn(net, imdb, @(x,y) getBatch(x,y), ...
                      'expDir', opts.expDir, ...
                      net.meta.trainOpts, ...
                      opts.train) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------
imdb = setupImdb(opts,net.meta,'test');
imdb.images.data = bsxfun(@minus, imdb.images.data, net.meta.normalization.averageImage);
cnn_test(net,imdb);

% -------------------------------------------------------------------------
function net = cnn_init(model,varargin)
% -------------------------------------------------------------------------
% CNN_IMAGENET_INIT  Initialize a standard CNN for ImageNet

opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.weightInitMethod = 'gaussian' ;
opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.dataDir = '../data';
opts = vl_argparse(opts, varargin) ;
nLabels = 2;

% Load the alexnet network to use as initialization for the weights.
% Keep the convolutional layers up to the 3rd max-pooling layer to reduce
% computational demands. The weights for these layers will remain frozen -
% only the weights of the newly introduced layers will be updated during
% training. You can also experiment with the case of updating weights 
% for these first layers as well during training.
net = your_code_goes_here;

% Meta parameters
net.meta.trainOpts.backPropDepth = select_how_deep_backpropagation_goes;
net.meta.outSize = [17,23]; % if you change the network architecture, you may have to adjust this
net.meta.normalization.imageSize = [286, 384. 3] ;
net.meta.inputSize = net.meta.normalization.imageSize ;
net.meta.normalization.interpolation = 'bilinear' ;
net.meta.normalization.averageImage = ...
    imresize(net.meta.normalization.averageImage, ...
    net.meta.normalization.imageSize(1:2), 'bilinear');

% Fill in default values
net = vl_simplenn_tidy(net) ;

% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad, init_bias)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{init_weight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0], ...
                           'opts', {convOpts}) ;
if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
                             'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), zeros(out, 2, 'single')}}, ...
                             'learningRate', [2 1 0.05], ...
                             'weightDecay', [0 0]) ;
end
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;

% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end

% --------------------------------------------------------------------
function net = add_norm(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'normalize', ...
                             'name', sprintf('norm%s', id), ...
                             'param', [5 1 0.0001/5 0.75]) ;
end

% --------------------------------------------------------------------
function net = add_dropout(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'dropout', ...
                             'name', sprintf('dropout%s', id), ...
                             'rate', 0.5) ;
end

% -------------------------------------------------------------------------
function [images,labels] = getBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(:,:,:,batch) ;

% --------------------------------------------------------------------
function [positions_full] = scan_points(filename)
% --------------------------------------------------------------------
fio = fopen(filename,'r');
for k=1:3, 
    fgetl(fio);  
end
for k=1:20
    line = fgetl(fio); 
    positions(:,k) = sscanf(line,'%f %f'); 
    
end 
fclose(fio);
positions_full  = positions(:,[1:4,15,5:14,16:end]);

% -------------------------------------------------------------------------
function [precision,recall,accuracy] = cnn_test(net, imdb)
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% Your code goes here
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
function imdb = setupImdb(opts, meta, set)
% -------------------------------------------------------------------------
facePath = fullfile(opts.dataDir, 'faces');
pointPath= fullfile(facePath, 'points_20');
faceImgs = dir(fullfile(facePath, '*.pgm'));
imsize   = meta.inputSize;
outSize  = meta.outSize;
switch set
    case 'train' % use odd indices for train set
        imgs = faceImgs(1:2:end);
    case 'test'  % and even indices for test set
        imgs = faceImgs(2:2:end);
    otherwise
        error('Set must be either ''train'' or ''test'' ')
end
nImgs = numel(imgs);
imdb.images.data   = zeros([imsize, nImgs], 'single');
imdb.images.labels = zeros([outSize(1:2), 1, nImgs], 'single'); 
for i=1:nImgs
    img   = imread(fullfile(facePath, imgs(i).name));
    pts   = scan_points(fullfile(pointPath, lower([imgs(i).name(1:end-4) '.pts'])));
    % Compute  bounding box for face
    x = round(pts(1,:)); y = round(pts(2,:));
    box = [max(1, min(x)), max(1, min(y)), min(imsize(2), max(x)), min(imsize(1), max(y))];
    
    seg = 2*ones(imsize(1:2), 'single'); % 1: face, 2: background
    seg(box(2):box(4), box(1):box(3)) = 1;
    % turn image from single-channel to RGB
    imdb.images.data(:,:,:,i) = repmat(img, [1 1 3]);
    % resize label maps (groundtruth) to the (subsampled) dimensions of the
    % network
    imdb.images.labels(:,:,:,i) = imresize(seg, outSize(1:2), 'nearest');
end
if strcmp(set,'train')
    % use the last 61 images as validation set (leaves 700 images for training)
    imdb.images.set = ones(1,nImgs);
    imdb.images.set(701:end) = 2; 
else
    % include the background images for testing
    backPath = fullfile(opts.dataDir, 'back');
    backImgs = dir(fullfile(backPath, '*jpg'));
    imdb.images.data(:,:,:,end+1:end+numel(backImgs))   = zeros([imsize, numel(backImgs)], 'single');
    imdb.images.labels(:,:,:,end+1:end+numel(backImgs)) = zeros([outSize(1:2), 1, numel(backImgs)], 'single'); 
    for i=1:numel(backImgs)
        img = imread(fullfile(backPath, backImgs(i).name));
        img = imresize(img, imsize(1:2), 'bilinear');
        % turn to gray scale and repmat for consistency
        imdb.images.data(:,:,:,nImgs+i) = repmat(rgb2gray(img), [1 1 3]);
    end
end
    