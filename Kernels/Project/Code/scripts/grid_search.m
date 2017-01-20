%------------------------------------------------------------------------------------
%% Main code Compute kernels, consider a kernels fusion and run SVM SMO per digit.
%------------------------------------------------------------------------------------
%% Settings:
clc; close all; clearvars;
set(0,'DefaultAxesFontSize',10)
set(0,'DefaultTextFontSize',10)
col=[
    9 125 18;
    97 6 158;
    199 2 45;
    2 2 214;
    237 99 7;
    250 120 202;
    145 145 145;
    127 245 163;
    252 215 5;
    50 155 247
]/255;
rng(17,'twister')
addpath (genpath('.'))
addpath(genpath('/usr/local/cellar/vlfeat')) % Change to only adding vl_tightsubplot
Xfull = importdata('Xtr.csv');
Yfull = importdata('Ytr.csv');
Yfull = Yfull.data(:,2);
val   = 1;                  % (0) to prepare the submission file
% ----------------------------------- Preprocessing parameters
global doviz
doviz              = 0;
PParams=[];
PParams.k          = 10*val;
PParams.Do_trim    = 1;
PParams.Do_deslant = 1;
PParams.Do_rotate  = 0; 
PParams.Do_scale   = 0;
PParams.angle      = 25;
PParams.rot        = .3; 
PParams.HOG        = 1;
PParams.cs         = 5;
PParams.sz         = [28 28];
npixels            = PParams.sz(1)*PParams.sz(2);
%-----------------------------------
[data] = preprocess(PParams,Xfull,Yfull);
%
Xs  = data.Xtr;  Ys  = data.Ytr;
Xvs = data.Xval; Yvs = data.Yval;

%% -------------------------------------------------  Solver parameters
%-------------------------- additional parameters per class in ovrSVM
SParams            = [];
SParams.tol        = 1e-5;
SParams.verb       = 0;
SParams.MKL        = 0;
% ------------------------------------------- Kernels parameters

KParamsRaw =[];
KParamsRaw.id = 'rbf';
KParamsRaw.sigma = 7;

KParamsHog1 =[];
KParamsHog1.id ='rbf';
KParamsHog1.sigma = 4;

KParamsHog2 =[];
KParamsHog2.id ='linear';
KParamsHog2.scale = 20;


fold = 1;
Xtr = Xs{fold}; Ytr = Ys{fold};
Xval = Xvs{fold}; Yval = Yvs{fold}; 
for s1 = [6 9 12 15]
    for s2 = [1 3 6 9 12]
        KParamsRaw.sigma = s1;
        KParamsHog1.sigma = s2;
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
        disp('-------------------------------------------------------');

        % ------------------------
        %  Training SVMs: 1 vs R
        % ------------------------
        for C_ = [.1 1 3 10 30]
            rng(123,'twister')
            C=C_*ones(10,1);
            for coeff_ = [.5 1 3 5]
                fprintf('C = %.2f \n Coeff = %.2f \n s1 = %d \n s2 = %d\n',C_,coeff_,s1,s2)  
                coeffs = coeff_*ones(10,1);
                [alpha, b, SV, perf] = ovrSVM(K, Ytr, C , coeffs ,SParams);
                scales = platt(alpha,SV,b,Ytr,K,perf);
                % Prediction
                %-----------------------(RAW pixels)
                KRv = kernel(Xtr(:,1:npixels),Xval(:,1:npixels), KParamsRaw);
                %-----------------------(HOG)
                KH1v  = kernel(Xtr(:,npixels+1:end),Xval(:,npixels+1:end), KParamsHog1);
                KH2v  = kernel(Xtr(:,npixels+1:end),Xval(:,npixels+1:end), KParamsHog2);
                %-----------------------(Fusion)
                Kv = fusion(KRv, KH1v, KH2v);
                [pred, votes] = ovrPred( alpha, SV, b, Ytr, Kv, scales);
                acc = sum(pred==Yval)/length(Yval); 
                fprintf('Accuracy (1 vs Rest): %.2f%%\n',acc*100);
                str = sprintf('Perf/C%dcoeff%ds1_%ds2_%d',floor(C_),floor(coeff_),s1,s2);
                figure(1); clf;
                Mis=show_misclassified(pred, Yval, Xval, 20,PParams.sz);
                drawnow,
                print('-f1',[str,'_mis'],'-djpeg')
        
                figure(2) ; clf;
                cm = cfmat(pred,Yval,0:9);
                title(sprintf('Overall accuracy %.2f%%',acc*100))
                drawnow,
                print('-f2',str,'-djpeg')
                
            end
        end  
    end 
end
   
   
