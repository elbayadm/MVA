% PART III: Towards large-scale retrieval

% setup MATLAB to use our software
setup ;

%% --------------------------------------------------------------------
%      Stage III.A: Accelerating descriptor matching with visual words
% --------------------------------------------------------------------

% Load a visual word vocabulary
load('data/oxbuild_lite_imdb_100k_ellipse_hessian.mat', 'vocab', 'kdtree') ;

% Load the two images
im1 = imread('data/oxbuild_lite/ashmolean_000007.jpg') ;
im2 = imread('data/oxbuild_lite/ashmolean_000028.jpg') ;

% Compute SIFT features for each
[frames1, descrs1] = getFeatures(im1, 'peakThreshold', 0.001, 'orientation', false) ;
[frames2, descrs2] = getFeatures(im2, 'peakThreshold', 0.001, 'orientation', false) ;

% Get the matches based on the raw descriptors
tic ;
[nn, dist2] = findNeighbours(descrs1, descrs2, 2) ;
nnThreshold = 0.8 ;
ratio2 = dist2(1,:) ./ dist2(2,:) ;
ok = ratio2 <= nnThreshold^2 ;
matches_raw = [find(ok) ; nn(1,ok)] ;
time_raw = toc ;

% Quantise the descritpors
tic;
words1 = vl_kdtreequery(kdtree, vocab, descrs1, 'maxNumComparisons', 1024) ;
mapping1=toc
words2 = vl_kdtreequery(kdtree, vocab, descrs2, 'maxNumComparisons', 1024) ;
mapping2=toc
% Get the matches based on the quantized descriptors
tic ;
matches_word_1 = matchWords(words1,words2,1) ;
time_word_1 = toc;

tic ;
matches_word_M = matchWords(words1,words2,size(words1,2)) ;
time_word_M = toc;

% Count inliers
inliers_raw = geometricVerification(frames1,frames2,matches_raw,'numRefinementIterations', 3) ;
inliers_word_1 = geometricVerification(frames1,frames2,matches_word_1,'numRefinementIterations', 3) ;
inliers_word_M = geometricVerification(frames1,frames2,matches_word_M,'numRefinementIterations', 3) ;

figure(1) ; clf ;
set(gcf,'name', 'III.B: Accelerating descriptor matching with visual words') ;

subplot(3,1,1) ; plotMatches(im1,im2,frames1,frames2,matches_raw(:,inliers_raw)) ;
title(sprintf('Verified matches on raw descritpors (%d in %.3g s from %d)',numel(inliers_raw),time_raw,numel(matches_raw))) ;

subplot(3,1,2) ; plotMatches(im1,im2,frames1,frames2,matches_word_1(:,inliers_word_1)) ;
title(sprintf('Verified matches on visual words (%d in %.3g s from %d)',numel(inliers_word_1),time_word_1,numel(matches_word_1))) ;

subplot(3,1,3) ; plotMatches(im1,im2,frames1,frames2,matches_word_M(:,inliers_word_M)) ;
title(sprintf('Verified matches on visual words (multi-matching) (%d in %.3g s from %d)',numel(inliers_word_M),time_word_M,numel(matches_word_M))) ;


%% --------------------------------------------------------------------
%                        Stage III.B: Searching with an inverted index
% --------------------------------------------------------------------

% Load an image DB
imdb = loadIndex('data/oxbuild_lite_imdb_100k_ellipse_hessian.mat') ;

% Compute a histogram for the query image
[h,frames,words] = getHistogramFromImage(imdb, im2) ;

% Score the other images by similarity to the query
tic ;
scores = h' * imdb.index ;
time_index = toc ;

% Plot results by decreasing score
figure(2) ; clf ;
plotRetrievedImages(imdb, scores, 'num', 25) ;
set(gcf,'name', 'III.B: Searching with an inverted index') ;
fprintf('Search time per database image: %.3g s\n', time_index / size(imdb.index,2)) ;

%% --------------------------------------------------------------------
%                                    Stage III.C: Geometric rearanking
% --------------------------------------------------------------------

% Rescore the top 16 images based on the number of
% inlier matches.

[~, perm] = sort(scores, 'descend') ;
for rank = 1:25
  matches = matchWords(words,imdb.images.words{perm(rank)}) ;
  inliers = geometricVerification(frames,imdb.images.frames{perm(rank)},...
                                  matches,'numRefinementIterations', 3) ;
  newScore = numel(inliers) ;
  scores(perm(rank)) = max(scores(perm(rank)), newScore) ;
end

% Plot results by decreasing score
figure(3) ; clf ;
plotRetrievedImages(imdb, scores, 'num', 25) ;
set(gcf,'name', 'III.B: Searching with an inverted index - verification') ;

%% --------------------------------------------------------------------
%                                             Stage III.D: Full system
% --------------------------------------------------------------------

% Load the database if not already in memory or if it is the one
% from exercise4
if ~exist('imdb', 'var') || isfield(imdb.images, 'wikiNames')
  imdb = loadIndex('data/oxbuild_lite_imdb_100k_ellipse_hessian.mat', ...
                   'sqrtHistograms', true) ;
end

% Search the database for a match to a given image. Note that URL
% can be a path to a file or a URL pointing to an image in the
% Internet.

url1 = 'data/queries/mistery-building1.jpg' ;
res = search(imdb, url1, 'box', []) ;

% Display the results
figure(4) ; clf ; set(gcf,'name', 'Part III.D: query image') ;
plotQueryImage(imdb, res) ;

figure(5) ; clf ; set(gcf,'name', 'Part III.D: search results') ;
plotRetrievedImages(imdb, res) ;
