% PART I: basic features

%% setup MATLAB to use our software
setup ;
%% --------------------------------------------------------------------
%                                   Stage I.A: SIFT features detection
% --------------------------------------------------------------------

% Load an image
im1 = imread('data/oxbuild_lite/all_souls_000002.jpg') ;
% Let the second image be a rotated and scaled version of the first
im3 = imresize(imrotate(im1,35,'bilinear'),0.7) ;
subplot(1,2,1) ; imagesc(im1) ; axis equal off ; hold on ;
subplot(1,2,2) ; imagesc(im3) ; axis equal off ;

%% Compute SIFT features for each
[frames1, descrs1] = getFeatures(im1, 'peakThreshold', 0.001) ;
[frames3, descrs3] = getFeatures(im3, 'peakThreshold', 0.001) ;

figure(1) ;
set(gcf,'name', 'Part I.A: SIFT features detection - synthetic pair') ;
subplot(1,2,1) ; imagesc(im1) ; axis equal off ; hold on ;
vl_plotframe(frames1, 'linewidth', 1) ;

subplot(1,2,2) ; imagesc(im3) ; axis equal off ; hold on ;
vl_plotframe(frames3, 'linewidth', 1) ;

%% Load a second image of the same scene
im2 = imread('data/oxbuild_lite/all_souls_000015.jpg') ;
[frames2, descrs2] = getFeatures(im2, 'peakThreshold', 0.001) ;

figure(2) ;
set(gcf,'name', 'Part I.A: SIFT features detection - real pair') ;
subplot(1,2,1) ; imagesc(im1) ; axis equal off ; hold on ;
vl_plotframe(frames1, 'linewidth', 1) ;

subplot(1,2,2) ; imagesc(im2) ; axis equal off ; hold on ;
vl_plotframe(frames2, 'linewidth', 1) ;

%% --------------------------------------------------------------------
%     Stage I.B: SIFT features descriptors and matching between images
% --------------------------------------------------------------------

% Visualize SIFT descriptors (only a few)
figure(3) ; clf ;
set(gcf,'name', 'Part I.B: SIFT descriptors') ;
imagesc(im1) ; axis equal off ;
vl_plotsiftdescriptor(descrs1(:,1:50:end), ...
                      frames1(:,1:50:end)) ;
hold on ;
vl_plotframe(frames1(:,1:50:end)) ;

%% Find for each desriptor in im1 the closest descriptor in im2
nn = findNeighbours(descrs1, descrs2) ;

% Construct a matrix of matches. Each column stores two index of
% matching features in im1 and im2
matches = [1:size(descrs1,2) ; nn(1,:)] ;

% Display the matches
figure(4) ; clf ;
set(gcf,'name', 'Part I.B: SIFT descriptors - matching') ;
plotMatches(im1,im2,frames1,frames2,matches(:,3:100:end)) ;
%title('Nearest neighbour matches') ;

%% --------------------------------------------------------------------
%  Stage I.C: Better matching (i) Lowe's second nearest neighbour test
% --------------------------------------------------------------------

% Find the top two neighbours as well as their distances
[nn, dist2] = findNeighbours(descrs1, descrs2, 2) ;

% Accept neighbours if their second best match is sufficiently far off
nnThreshold = .8 ;
ratio2 = dist2(1,:) ./ dist2(2,:) ;
ok = ratio2 <= nnThreshold^2 ;

% Construct a list of filtered matches
matches_2nn = [find(ok) ; nn(1, ok)] ;

% Alternatively, do not do the second nearest neighbourhod test.
% Instead, match each feature to its two closest neighbours and let
% the geometric verification step figure it out.

% matches_2nn = [1:size(nn,2), 1:size(nn,2) ; nn(1,:), nn(2,:)] ;

% Display the matches
figure(5) ; clf ;
set(gcf,'name', 'Part I.C: SIFT descriptors - Lowe''s test') ;
plotMatches(im1,im2,frames1,frames2,matches_2nn) ;
%% --------------------------------------------------------------------
%             Stage I.D: Better matching (ii) geometric transformation
% --------------------------------------------------------------------
%Voting the transformatons:
%The bins structure:
Corresp_theta=(-2*pi:.1:2*pi);
Corresp_s=.1*(1:20);
Corresp_x=.5*size(im1,1)*(-4:.5:4);
Corresp_y=.5*size(im2,2)*(-4:.5:4);
H=zeros(length(Corresp_s)-1,length(Corresp_theta)-1,length(Corresp_x)-1,length(Corresp_y)-1);

for index = 1:size(matches_2nn,2)
    frm1=frames1(:,matches_2nn(1,index));
    frm2=frames2(:,matches_2nn(2,index));
    %the transformation:
    theta=frm2(4)-frm1(4);
    tx=frm2(1)-frm1(1);
    ty=frm2(2)-frm1(2);
    s=frm2(3)/frm1(3);
    %Discretization and voting:
    thi=find(Corresp_theta-theta>0,1,'first')-1;
    si=find(Corresp_s-s>0,1,'first')-1;
    txi=find(Corresp_x-tx>0,1,'first')-1;
    tyi=find(Corresp_y-ty>0,1,'first')-1;
    H(si,thi,txi,tyi)=H(si,thi,txi,tyi)+1;
end
labels_s=zeros(length(Corresp_s)-1,1);
for i=1:length(Corresp_s)-1
    labels_s(i)=(Corresp_s(i)+Corresp_s(i+1))/2;
end
labels_theta=zeros(length(Corresp_theta)-1,1);
for i=1:length(Corresp_theta)-1
    labels_theta(i)=(Corresp_theta(i)+Corresp_theta(i+1))/2;
end
labels_x=zeros(length(Corresp_x)-1,1);
for i=1:length(Corresp_x)-1
    labels_x(i)=(Corresp_x(i)+Corresp_x(i+1))/2;
end
labels_y=zeros(length(Corresp_y)-1,1);
for i=1:length(Corresp_y)-1
    labels_y(i)=(Corresp_y(i)+Corresp_y(i+1))/2;
end
figure,
subplot(2,1,1)
%Scales histogram:
bar(labels_s,sum(sum(sum(H,2),3),4))
title('Histogram of scales s')

subplot(2,1,2)
%Angles histogram:
bar(labels_theta,sum(sum(sum(H,1),3),4))
title('Histogram of \theta')

figure,
subplot(2,1,1)
%X histogram:
bar(labels_x,permute(sum(sum(sum(H,4),1),2),[3,1,2]))
title('Histogram of t_x')

subplot(2,1,2)
%Y histogram:
bar(labels_y,permute(sum(sum(sum(H,3),1),2),[4,3,1,2]))
title('Histogram of t_y')
[s,theta,tx,ty]=ind2sub(size(H),find(H>2));


%% Geometric verification:
inliers = geometricVerification(frames1, frames2, matches_2nn, 'numRefinementIterations', 8) ;
inliers1 = geometricVerification(frames1, frames2, matches_2nn, 'numRefinementIterations', 8,'tolerance2', 50);
inliers2 = geometricVerification(frames1, frames2, matches_2nn, 'numRefinementIterations', 8,'tolerance3', 50);
length(inliers)
length(inliers1)
length(inliers2)

matches_geo = matches_2nn(:, inliers) ;
A=setdiff(inliers,inliers1);
B=setdiff(inliers2,inliers);
% Display the matches
figure(6) ; clf ;
set(gcf,'name', 'Part I.D: SIFT descriptors - geometric verification') ;
plotMatches(im1,im2,frames1,frames2,matches_geo);
figure,
plotMatches(im1,im2,frames1,frames2,matches_2nn(:, A));
title(sprintf('Eliminated %d matches - tolerance 2 - total =%d',length(A),length(inliers1))) ;
figure,
plotMatches(im1,im2,frames1,frames2,matches_2nn(:, B));
title(sprintf('Extra %d matches - tolerance 3 - total =%d',length(B),length(inliers2))) ;
%% Geometric verification witouth 2NN rule:
% Finding the three nearest neighbors
nn_123 = findNeighbours(descrs1, descrs2, 3);
%1NN directly:
matches_1=[1:size(nn_123,2) ; nn_123(1,:)];
%1NN and 2NN:
matches_12 = [1:size(nn_123,2), 1:size(nn_123,2) ; nn_123(1,:), nn_123(2,:)];
%1NN, 2NN and 3NN: 
matches_123 = [1:size(nn_123,2), 1:size(nn_123,2), 1:size(nn_123,2) ; nn_123(1,:), nn_123(2,:), nn_123(3,:)];

length(matches_1)
length(matches_12)
length(matches_123)
% Geometric verififcation:
tic;
inliers_1 = geometricVerification(frames1, frames2, matches_1, 'numRefinementIterations', 1) ;
NN1=toc;
tic;
inliers_12 = geometricVerification(frames1, frames2, matches_12, 'numRefinementIterations', 1) ;
NN2=toc;
tic;
inliers_123 = geometricVerification(frames1, frames2, matches_123, 'numRefinementIterations', 1) ;
NN3=toc;
length(inliers_1)
length(inliers_12)
length(inliers_123)


figure,
plotMatches(im1,im2,frames1,frames2,matches_1(:,inliers_1));
title(sprintf('%d matches - 1NN - runtime %.4f s',length(inliers_1),NN1)) ;

figure,
plotMatches(im1,im2,frames1,frames2,matches_12(:,inliers_12));
title(sprintf('%d matches - 1NN , 2NN - runtime %.4f s',length(inliers_12),NN2)) ;

figure,
plotMatches(im1,im2,frames1,frames2,matches_123(:,inliers_123));
title(sprintf('%d matches - 1NN,2NN,3NN - runtime %.4f s',length(inliers_123),NN3)) ;

