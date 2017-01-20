% Process matrix saved from getLmdb.py
images = importdata('images.mat');
n=length(images);
Data=zeros(n/3,96,96,3,'uint8');
j=1;
textprogressbar('Processing the images ');
for i=1:3:n
    textprogressbar(i,n/3)
    im = squeeze(images(i,:,:,:));
    im = permute(im,[2,3,1]);
    im = cat(3,im(:,:,3),im(:,:,2),im(:,:,1));
    Data(j,:,:,:)=im;
    j=j+1;
end
textprogressbar('done')
save('new_images.mat','Data')