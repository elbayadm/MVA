%------------------------------------------------------------------------------------
% Main script:
% 1 .Compute the chosen kernels
% 2 . consider a kernels fusion.
% 3. run SVM SMO per digit (one vs rest).
% 4. Learn platt's coefficients
% 5. Predict
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
val   = 0;   %----------------------- set to 0 for submission
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
%--------------------------
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
%%
for fold = 1:length(Xs)

    Xtr = Xs{fold}; Ytr = Ys{fold};
    if val
        Xval = Xvs{fold}; Yval = Yvs{fold}; 
    end
    % -----------------------
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
    %  Training SVMs: 1 vs R
    % ------------------------
    
    str='model1';
    rng(123,'twister')
  
    C=10*ones(10,1);
    coeffs =2*ones(10,1);
    [alpha, b, SV, perf] = ovrSVM(K, Ytr, C , coeffs ,SParams);
    scales = platt(alpha,SV,b,Ytr,K,perf);
    
    if val
        % Prediction
        %-----------------------(RAW pixels)
        KRv = kernel(Xtr(:,1:npixels),Xval(:,1:npixels), KParamsRaw);
        %-----------------------(HOG)
        KH1v  = kernel(Xtr(:,npixels+1:end),Xval(:,npixels+1:end), KParamsHog1);
        KH2v  = kernel(Xtr(:,npixels+1:end),Xval(:,npixels+1:end), KParamsHog2);
        %-----------------------(Fusion)
        Kv = fusion(KRv, KH1v, KH2v);
        [pred, votes] = ovrPred( alpha, SV, b, Ytr, Kv, scales);
        acc(end+1) = sum(pred==Yval)/length(Yval); 
        fprintf('Accuracy (1 vs Rest): %.2f%%\n',acc(end)*100);
        figure(1); clf;
        Mis=show_misclassified(pred, Yval, Xval, 20,PParams.sz);
        drawnow,
        
        figure(2) ; clf;
        cm = cfmat(pred,Yval,0:9);
        drawnow,
        if fold==1
            lg  = fopen([str '.log'],'w');
            figure(3); clf;
            legs={};
            for i=1:length(perf)
                fprintf(lg,'[%d] C_pos= %#5.2f - C_neg= %#5.2f - |SV|= %4i - |bSV|= %4i - iter= %4i - |pos|= %4i\n',...
                          i-1,perf{i}.Cpos,perf{i}.Cneg,perf{i}.nSV,perf{i}.bSV,perf{i}.iter, perf{i}.pos); 
                plot(1:length(perf{i}.obj),perf{i}.obj,'linewidth',2,'color',col(i,:))
                legs{end+1} = perf{i}.id;
                hold on,
            end
            title('Objective optimization per class','fontsize',16)
            legend(legs,'location','northwest');
            xlim([0 1e4])
            fprintf(lg,'Accuracy (1 vs Rest): %.2f%%\n',acc(end)*100);
            print('-f1',[str,'_mis'],'-dpdf')
            print('-f2',str,'-dpdf')
            print('-f3',[str,'_obj'],'-dpdf')
            save ([str,'.mat'],'PParams','SParams','KParamsHog1','KParamsHog2', 'KParamsRaw','fusion', 'perf')
        end 
    else
        % Final output
        Xtest = importdata('Xte.csv');
        Xtest = preprocess(PParams,Xtest);
        % -----------------------(RAW pixels)
        disp('Pixels kernel')
        KRt = kernel(Xtr(:,1:npixels),Xtest(:,1:npixels), KParamsRaw);
        %-----------------------(HOG)
        disp('HOG kernel (1)');
        KH1t  = kernel(Xtr(:,npixels+1:end),Xtest(:,npixels+1:end), KParamsHog1);
        disp('HOG kernel (2)');
        KH2t  = kernel(Xtr(:,npixels+1:end),Xtest(:,npixels+1:end), KParamsHog2);
        %-----------------------(Fusion)
        Kt = fusion(KRt, KH1t, KH2t);
        [pred, ~] = ovrPred( alpha, SV, b, Ytr, Kt, scales);
        % Print the output file:
        pred = [(1:length(pred))', pred];
        out = fopen('submission.csv','w');
        fprintf(out,'Id,Prediction\n');
        fprintf(out,'%d,%d\n',pred');
        fclose(out);
        disp('Prediction completed.')
    end
end

