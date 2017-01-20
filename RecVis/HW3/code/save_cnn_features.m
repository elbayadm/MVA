%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Saving mat file of the CNN output {softmax, fc8 and fc9} for each category
%    of interest - training and validation set
%    + background images as well.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net=load('imagenet-vgg-f.mat');
%%
%category = 'motorbike' ;
%category = 'aeroplane' ;
category = 'person' ;

%% Category training images
hfile = fopen(['data/' category '_train.txt']);
C = textscan(hfile,'%s'); C=C{1};
p1=length(net.layers{18}.weights{1,2});
p2=length(net.layers{20}.weights{1,2});
p3=length(net.classes.name);
fc7=zeros(p1,length(C));
fc8=zeros(p2,length(C));
softmax=zeros(p3,length(C));
for i=1:length(C)
    im =imread(['HW2/code/data/images/',C{i},'.jpg']);
    im_ = single(im);
    im_ = imresize(im_, net.normalization.imageSize(1:2));
    im_ = im_ - net.normalization.averageImage;
    res = vl_simplenn(net, im_);
    fc7(:,i)=res(19).x(:); %fc7 = layer 18
    fc8(:,i)=res(21).x(:); %fc8 = layer 20
    softmax(:,i)=res(22).x(:); %softmax = layer 21 % The last
end
names=C';
save(['data/' category '_train_cnn.mat'],'fc7','fc8','softmax','names');
clear softmax fc7 fc8 C names;
fprintf(2,'Done\n');
%% Category validation images:
hfile = fopen(['data/' category '_val.txt']);
C = textscan(hfile,'%s'); C=C{1};
p=length(net.layers{18}.weights{1,2})+length(net.layers{20}.weights{1,2})+length(net.classes.name);
p1=length(net.layers{18}.weights{1,2});
p2=length(net.layers{20}.weights{1,2});
p3=length(net.classes.name);
fc7=zeros(p1,length(C));
fc8=zeros(p2,length(C));
softmax=zeros(p3,length(C));
for i=1:length(C)
    im =imread(['HW2/code/data/images/',C{i},'.jpg']);
    im_ = single(im);
    im_ = imresize(im_, net.normalization.imageSize(1:2));
    im_ = im_ - net.normalization.averageImage;
    res = vl_simplenn(net, im_);
    fc7(:,i)=res(19).x(:); %fc7 = layer 18
    fc8(:,i)=res(21).x(:); %fc8 = layer 20
    softmax(:,i)=res(22).x(:); %softmax = layer 21 % The lastend
end
names=C';
save(['data/' category '_val_cnn.mat'],'fc7','fc8','softmax','names');
clear softmax fc7 fc8 C names;
fprintf(2,'Done\n');
%% Training background images
hfile = fopen('data/background_train.txt');
C = textscan(hfile,'%s'); C=C{1};
p1=length(net.layers{18}.weights{1,2});
p2=length(net.layers{20}.weights{1,2});
p3=length(net.classes.name);
fc7=zeros(p1,length(C));
fc8=zeros(p2,length(C));
softmax=zeros(p3,length(C));
for i=1:length(C)
    im =imread(['HW2/code/data/images/',C{i},'.jpg']);
    im_ = single(im);
    im_ = imresize(im_, net.normalization.imageSize(1:2));
    im_ = im_ - net.normalization.averageImage;
    res = vl_simplenn(net, im_);
    fc7(:,i)=res(19).x(:); %fc7 = layer 18
    fc8(:,i)=res(21).x(:); %fc8 = layer 20
    softmax(:,i)=res(22).x(:); %softmax = layer 21 % The lastend
end
names=C';
save('data/background_train_cnn.mat','fc7','fc8','softmax','names');
clear softmax fc7 fc8 C names;
fprintf(2,'Done\n');
%% Validation background images
hfile = fopen('data/background_val.txt');
C = textscan(hfile,'%s'); C=C{1};
p1=length(net.layers{18}.weights{1,2});
p2=length(net.layers{20}.weights{1,2});
p3=length(net.classes.name);
fc7=zeros(p1,length(C));
fc8=zeros(p2,length(C));
softmax=zeros(p3,length(C));
for i=1:length(C)
    im =imread(['HW2/code/data/images/',C{i},'.jpg']);
    im_ = single(im);
    im_ = imresize(im_, net.normalization.imageSize(1:2));
    im_ = im_ - net.normalization.averageImage;
    res = vl_simplenn(net, im_);
    fc7(:,i)=res(19).x(:); %fc7 = layer 18
    fc8(:,i)=res(21).x(:); %fc8 = layer 20
    softmax(:,i)=res(22).x(:); %softmax = layer 21 % The lastend
end
names=C';
save('data/background_val_cnn.mat','fc7','fc8','softmax','names');
clear softmax fc7 fc8 C names;
fprintf(2,'Done\n');
