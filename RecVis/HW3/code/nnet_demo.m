% Demo script for training two layer fully connected neural network 
% with one hidden layer
% 
% Josef.Sivic@ens.fr
% Adapted from Nicolas Le Roux

clc; close all; clear; % clear all current variables.
doviz=0; %Disable visualization
doprint=1; % printing the figures to pdf
verify=0; %run the gradient verification

% load training data
tmp = load('double_moon_train1000.mat');
Xtr = tmp.X';
Ytr = tmp.Y';

% load the validation data
tmp = load('double_moon_val1000.mat');
Xval = tmp.X';
Yval = tmp.Y';

% train fully connected neural network with 1 hidden layer
% number of hidden units, i.e. the dimensionality of the hidden layer
%h=1;
%h=2;
%h=5;
%h=7;
%h=10;
h=100;

di = 2; % input dimension (2D) -- do not change
do = 1; % output dimension (1D - classification) -- do not change

%lrate     = 2;
%lrate     = 0.2;
lrate     = 0.02;
%lrate     = 0.002;

nsamples  = length(Ytr);
visualization_step = 1000; % visualize output only these steps

% randomly initialize parameters of the model
%rng(6883,'V4');% to ensure we get reproducible results -if needed 

Wi = rand(h,di);
bi = rand(h,1);
Wo = rand(1,h);
bo = rand(1,1);

% Verify the derivatives:
if verify
    %Pick a sample at random:
    if exist('grad_verif.txt', 'file')==2
      delete('grad_verif.txt');
    end
    diary('grad_verif.txt')
    diary on;

    n = randi(nsamples);
    X = Xtr(:,n);
    Y = Ytr(:,n);
    epsilon=.0001;
    [grad_s_Wi_approx, grad_s_Wo_approx, grad_s_bi_approx, grad_s_bo_approx] = gradient_nn_approx(X,Y,Wi,bi,Wo,bo,epsilon)
    [grad_s_Wi, grad_s_Wo, grad_s_bi, grad_s_bo] =gradient_nn(X,Y,Wi,bi,Wo,bo)

    %Test:
    fprintf('For X=[%.4f,%.4f] of label Y=%d\n',X,Y);
    fprintf('Gradient etimation error on bo= %.3e\n',norm(grad_s_bo-grad_s_bo_approx));
    fprintf('Gradient etimation error on Wo= %.3e\n',norm(grad_s_Wo-grad_s_Wo_approx));
    fprintf('Gradient etimation error on bi= %.3e\n',norm(grad_s_bi-grad_s_bi_approx));
    fprintf('Gradient etimation error on Wi= %.3e\n',norm(grad_s_Wi-grad_s_Wi_approx));
    diary off;
end

% Train the NN
iterations=100*nsamples;
for i = 1:iterations % hundred passes through the data
    
    % draw an example at random
    n = randi(nsamples);
    
    X = Xtr(:,n);
    Y = Ytr(:,n); % desired output
    
    % compute gradient 
    [grad_s_Wi, grad_s_Wo, grad_s_bi, grad_s_bo] = ... 
                                gradient_nn(X,Y,Wi,bi,Wo,bo);

    % gradient update                                 
    Wi = Wi - lrate.*grad_s_Wi;
    Wo = Wo - lrate.*grad_s_Wo;
    bi = bi - lrate.*grad_s_bi;
    bo = bo - lrate.*grad_s_bo;
    
    % plot training error
    [Po,Yo,loss]    = nnet_forward_logloss(Xtr,Ytr,Wi,bi,Wo,bo);
    Yo_class        = sign(Yo);
    tr_error(i)     = sum((Yo_class - Ytr)~=0)./length(Ytr);
    
    % plot validation error
    [Pov,Yov,lossv]    = nnet_forward_logloss(Xval,Yval,Wi,bi,Wo,bo);
    Yov_class          = sign(Yov);
    val_error(i)       = sum((Yov_class - Yval)~=0)./length(Yval);
    
   
    % visualization only every visualiztion_step-th iteration       
    if doviz
        if ~mod(i,visualization_step) 

            % show decision boundary
            plot_decision_boundary(Xtr,Ytr,Wi,bi,Wo,bo);

            % plot the evolution of the training and test errors
            figure(3); hold off;
            plot(tr_error);
            title(sprintf('Training error: %.2f %%',tr_error(i)*100));
            grid on;

            figure(4); hold off;
            plot(val_error);
            title(sprintf('Validation error: %.2f %%',val_error(i)*100));
            grid on;
        end
    end;
    
%     if ~mod(i,10000)
%         fprintf('Running %d%%:\n',i/nsamples);
%         fprintf('Training error=%.2f%%\n',tr_error(i)*100);
%         fprintf('Validation error=%.2f%%\n',val_error(i)*100);
%         fprintf('-----------------------\n');
%     end
    
end;
%fprintf('Done\n');
%
fprintf('Case h=%d, lrate=%.5f\n',h,lrate);
fprintf('Training error=%.2f%%\n',tr_error(end)*100);
fprintf('Validation error=%.2f%%\n',val_error(end)*100);
Tr_cvg=find(tr_error,1,'last')+1;
if Tr_cvg<=iterations
    fprintf('Training Cvg at i=%d\n',Tr_cvg);
else
    fprintf('Divergent\n');
end
plot_decision_boundary(Xtr,Ytr,Wi,bi,Wo,bo);
if doprint
print(1,'-dpdf',sprintf('images/tocrop/DB_h%d.pdf',h), '-opengl')
end
figure(2);
plot(tr_error);
grid on;
xlabel('iteration','fontsize',20);
ylabel('Error','fontsize',20);
if Tr_cvg<=iterations
    text(1/5*iterations,.3,sprintf('Convergence at iter=%d',Tr_cvg),'fontsize',20,'color',[0 .6 .2]);
else
    text(4e4,.3,'Divergent','color','r','fontsize',20)
end
if doprint 
    print(2,'-dpdf',sprintf('images/tocrop/tr_error_h%d.pdf',h), '-opengl')
end

figure(3);
plot(val_error');
grid on;
xlabel('iteration','fontsize',20);
ylabel('Error','fontsize',20);
% if doprint
%     print(3,'-dpdf','images/tocrop/val_error.pdf', '-opengl')
% end
