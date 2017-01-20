%% Settings:
warning 'off'
clearvars; clc; close all;
addpath(genpath('Code'))
addpath(genpath('/usr/local/cellar/matconvnet/matlab'))
addpath(genpath('Models'))
addpath(genpath('Data'))
addpath(genpath('figures'))
vl_setupnn

%% SimpleNN
% Could be loaded as Dag eventually
%------------------------------------------------------------------------VGG deep
 net = load('imagenet-vgg-verydeep-16.mat');
% Display the net architecture
% vl_simplenn_display(net)

%% DagNN
%------------------------------------------------------------------------ Imagenet
% net=dagnn.DagNN.loadobj(load('imagenet-vgg-verydeep-16.mat'));
% net = dagnn.DagNN.loadobj(load('imagenet-matconvnet-alex.mat')) ; 
%  net = dagnn.DagNN.loadobj(load('bvlc_googlenet_dag.mat')) ;
%------------------------------------------------------------------------ Places205
% net = dagnn.DagNN.loadobj(load('places205_googlenet_dag.mat'));

%% List layers (DagNN)
clc;
for ii=1:length(net.layers)
    fprintf('%d : %s (%s)\n',ii,net.layers(ii).name,class(net.layers(ii).block));
end

%% Classify an image:
% load and preprocess an image
 im = imread('jurassic.jpg') ;
 im_=single(im);
 im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
%recent models use a 1x1 average image (i.e. just an average color)
 im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage);

if ~isstruct(net.layers)
	% run the CNN
	n = length(net.layers);   % Number of layers
	layer=n;                  % Picking the last layer
	res = init_res(layer) ;       % initializate response structure
	res(1).x = single(im_);               % load image in first layer
	res = forwardto_nn(net, layer, res) ; % Forward propagation to selected layer

	% show the classification result
	scores = squeeze(gather(res(layer+1).x)) ;
else
	net.eval({net.vars(1).name,im_})
	scores = net.vars(net.getVarIndex(net.vars(end).name)).value ;
	scores = squeeze(gather(scores)) ;
end

[bestScore, best] = max(scores) ;
figure(1) ; clf ; imshow(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;

%% Deepdreaming:  Global parameters
    close all; clc;
    global debug;  debug=0;
    global isDag;  isDag=isstruct(net.layers);
    global sketch; sketch=0;
    global mean_image,
    if isfield(net,'meta')
        mean_image=net.meta.normalization.averageImage;
        if size(mean_image,1)>1
            mean_image=mean(mean(mean_image,1),2);
        end
    else
        mean_image=zeros(1,1,3);
    end
    
opts=setopts();

opts.step=3;
opts.iters=10;

opts.crop=0;
opts.octave=0;
opts.layer= 13;  

% % Controlling dreams:
% guide = imread('sky.jpg');
% opts.guide=im2single(guide);
% opts.objective='guide';

% %Multi layers
% opts.objective='sumN2';
% opts.combine = [.5 .5];

% % Single unit
% opts.objective='neuron';
% opts.activate=954;

opts.fig='sky';
opts.net='googlenet_places';


% Start from noise
% im=randn(224,224,3);

% Load the image
%im = imread('sky.jpg');

%im=imresize(im,[224 224]);
im=im2single(im);
dreams=dream(net, im, opts);
