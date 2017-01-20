run('../matlab/vl_setupnn.m');
load('dataset_without_order_info_256.mat');
load('model_without_order_info_256.mat');

testX = 255-imdb.images.data(:,:,:,imdb.images.set==3);
testY = imdb.images.labels(:,imdb.images.set==3);
predY = zeros(1,length(testY));

for i = 1:length(testY)
    i 
    testimg = testX(:,:,:,i);
    im1 = testimg(1:225,1:225) ;
    im2 = testimg(32:256,1:225) ;
    im3 = testimg(1:225,32:256) ;
    im4 = testimg(32:256,32:256) ;
    im5 = testimg(16:240,16:240) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    im_new = cat(4,im15,fliplr(im15));
    net.layers{end}.class = kron(ones(1,size(im_new,4)),testY(i));
    res = vl_simplenn(net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    pred = squeeze(gather(res(end-1).x));
    pred = sum(pred,2);
    [~,pred] = max(pred);
    predY(i) = pred;
end

mean(predY==testY)
