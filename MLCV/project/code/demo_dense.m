
%% addpaths
this_dir    = fileparts(mfilename('fullpath')); addpath(this_dir); %addpath(fullfile(this_dir,'steerable'));
input_image = imread('1.jpg');
input_image_gray  = single(rgb2gray(imread('1.jpg')));

%%---------------------------------------------------------
%% Part 1 
%% dense sift features
%%---------------------------------------------------------
[fts,crds,idxs] =  get_features(input_image_gray,'sift');

%% choose a single image point & visualize sift descriptor around it  
point = 100000;
figure(1),imshow(input_image_gray/255);
hold on,scatter(crds(1,point),crds(2,point));
vl_plotsiftdescriptor(fts(1:128,point),[crds(:,point);4;0]);
pause;
clf;

%% show elements of sift descriptor, 'dense mode'
[sv,sh] = size(input_image_gray);

figure,
dims_wt = [1:4]; %% sift dimensions being visualized (out of 128)
cnt = 0;
for dim = [dims_wt]
    response            = zeros(sv,sh);
    response(idxs)      = fts(dim,:);
    cnt = cnt  + 1;
    subplot(2,2,cnt);
    imshow(response,[]); title(sprintf('dimension %i',dim));
end

figure,
dims_wt = [125:128]; %% sift dimensions being visualized (out of 128)
cnt = 0;
for dim = [dims_wt]
    response            = zeros(sv,sh);
    response(idxs)      = fts(dim,:);
    cnt = cnt  + 1;
    subplot(2,2,cnt);
    imshow(response,[]);  title(sprintf('dimension %i',dim));
end

%%---------------------------------------------------------
%% Part 2
%% CNN features
%%---------------------------------------------------------

% Load pre-trained CNN model and discard the fully-connected layers 
% (we only use convolutional layers to extract features). We test results 
% using either conv2 or conv5 features. 
% cnn_net = load('/home/tsogkas/code/matconvnet/data/models/imagenet-caffe-ref.mat'); 
cnn_net = load('/home/tsogkas/Desktop/imagenet-caffe-ref.mat'); 
cnn_net.layers = cnn_net.layers(1:14);

figure,imshow(input_image);
%% Extract and visualize features
[cnn_fts, crds, idxs] = get_features(input_image,'cnn5',[],cnn_net);
[sv,sh,~] = size(input_image);

figure,
dims_wt = [1:4]; %% cnn dimensions being visualized (out of 256)
cnt = 0;
for dim = [dims_wt]
    response       = zeros(sv,sh);
    response(idxs) = cnn_fts(dim,:);
    cnt = cnt  + 1;
    subplot(2,2,cnt);
    imshow(response,[]); title(sprintf('dimension %i',dim));
end

figure,
dims_wt = [253:256]; %% sift dimensions being visualized (out of 256)
cnt = 0;
for dim = [dims_wt]
    response       = zeros(sv,sh);
    response(idxs) = cnn_fts(dim,:);
    cnt = cnt  + 1;
    subplot(2,2,cnt);
    imshow(response,[]);  title(sprintf('dimension %i',dim));
end
