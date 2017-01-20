%------------------------------------------------------------------------------------
% Main script:
% 1 .Compute the chosen kernels
% 2 . consider a kernels fusion.
% 3. run SVM SMO per digit (one vs one).
% 5. Predict with voting
%------------------------------------------------------------------------------------
% Settings:
clc; close all; clear all;
set(0,'DefaultAxesFontSize',10)
set(0,'DefaultTextFontSize',10)
col=[9 125 18;97 6 158;199 2 45;2 2 214;237 99 7;
    250 120 202;145 145 145;127 245 163;252 215 5;
    50 155 247]/255;
rng(17,'twister')
addpath (genpath('.'))
Xfull = importdata('Xtr.csv');
Yfull = importdata('Ytr.csv');
Yfull = Yfull.data(:,2);
val   = 1;   %----------------------- set to 0 for submission
% ----------------------------------- Preprocessing parameters
global doviz %----------------------- visualize random samples 
doviz              = 1;
PParams            = [];
PParams.k          = 5*val;  % folds for CV
PParams.do_trim    = 1;      % trim the images (remove the black borders)
PParams.do_deslant = 1;      % perform slant removal
PParams.do_blur    = 1;      % perform gaussian blurring
PParams.do_median  = 0;      % perform median filtering
PParams.do_rotate  = 0;      % add rotated samples
PParams.angle      = 25;     % range of rotation angles
PParams.rot        = .3;     % ratio of rotated images
PParams.do_norm    = 0;      % normalize the features   
PParams.hog        = 1;
PParams.cs         = 5;
PParams.sz         = [28 28];
npixels            = PParams.sz(1)*PParams.sz(2);
%-----------------------------------
[data] = preprocess(PParams,Xfull,Yfull);
Xs  = data.Xtr;  Ys  = data.Ytr;
Xvs = data.Xval; Yvs = data.Yval;
% Solver parameters
%-------------------------- additional parameters per class in ovrSVM
SParams            = [];
SParams.tol        = 1e-5;  % optimization tolerance
SParams.verb       = 1;     % optimization verbosity
SParams.MKL        = 0;     % 1 to use MKL algorithm

% Kernels parameters
% --------------------------
% Pixels kernel:
KParamsRaw =[];
KParamsRaw.id = 'rbf';
KParamsRaw.sigma = 7;

% 1st Hog kernels
KParamsHog1 =[];
KParamsHog1.id ='rbf';
KParamsHog1.sigma = 4;
KParamsHog1.scale = 50;
KParamsHog1.gamma = .01;

% 2nd Hog kernels
KParamsHog2 =[];
KParamsHog2.id ='linear';
KParamsHog2.scale = 20;


acc = [];
for fold = 1:length(Xs)
    Xtr = Xs{fold}; Ytr = Ys{fold};
    if val
        Xval = Xvs{fold}; Yval = Yvs{fold}; 
    end
    disp('-------------------------------------------------------');
    fprintf 'Computing the full kernel matrix..\n'
    tic,
    %-----------------------(RAW pixels)
    disp('Pixels kernel')
    KR = kernel(Xtr(:,1:npixels),Xtr(:,1:npixels), KParamsRaw);
    %-----------------------(HOG)
    disp('HOG kernel (1)');
    KH1  = kernel(Xtr(:,npixels+1:end),Xtr(:,npixels+1:end), KParamsHog1);
    disp('HOG kernel (2)');
    KH2  = kernel(Xtr(:,npixels+1:end),Xtr(:,npixels+1:end), KParamsHog2);
    disp('Done')
     % -----------------------(Fusion)
    fusion = @(KR, KH1, KH2) KR.*(KH1 + KH2 + KH1.*KH2);
    K = fusion(KR,KH1,KH2);
    kt      = toc;
    temp    = whos('K');
    fprintf('Kernel matrix : size %.1fMb - computation time %.2fs\n',temp.bytes/1e6,kt);
    disp('-------------------------------------------------------');

    % ------------------------
    %  Training SVMs: 1 vs 1
    % ------------------------
    rng(123,'twister')
    C = 10;
    coeff = 1;
    [pairs, idx, alpha, b, SV, perf] = ovoSVM(K, Ytr,C, SParams); %p for pairs
    %%
    if val
        % Prediction: - Votes
        %-----------------------(RAW pixels)
        KRv = kernel(Xtr(:,1:npixels),Xval(:,1:npixels), KParamsRaw);
        %-----------------------(HOG)
        KH1v  = kernel(Xtr(:,npixels+1:end),Xval(:,npixels+1:end), KParamsHog1);
        KH2v  = kernel(Xtr(:,npixels+1:end),Xval(:,npixels+1:end), KParamsHog2);
        %-----------------------(Fusion)
        Kv = fusion(KRv, KH1v, KH2v);
        [pred] = ovoPred( alpha, SV, b, Ytr, Kv, pairs, idx);
        acc(end+1) = sum(pred==Yval)/length(Yval);
        fprintf('Accuracy (1 vs 1): %.2f%%\n',acc(end)*100);
        figure(1); clf;
        Mis=show_misclassified(pred, Yval, Xval, 20,PParams.sz);
        drawnow,
        figure(3) ; clf;
        cm=cfmat(pred,Yval,0:9);
        drawnow,
        for p=1:length(pairs)
            i=pairs(p,1)+1; j= pairs(p,2)+1;
            fprintf('[%s] C_pos= %#5.2f - C_neg= %#5.2f - |SV|= %4i - |bSV|= %4i - iter= %4i - |pos|= %4i\n',...
                      perf{i}{j}.id,perf{i}{j}.Cpos,perf{i}{j}.Cneg,perf{i}{j}.nSV,perf{i}{j}.bSV,perf{i}{j}.iter, perf{i}{j}.pos);
        end
    end
    
end
% One to one accuracy:
idxv = {};
for i=1:length(pairs)
    idxv{pairs(i,1)+1}{pairs(i,2)+1} = union(find(Yval==pairs(i,1)),find(Yval==pairs(i,2)));
end
acc1o1 = ovoAccuracy(pairs,alpha,b,SV,idx,idxv, Kv, Ytr, Yval)
