warning off all;
disp('Training the gist model')

Mtest = length(test_imdb);

class_training = -ones([M N]);
class_test  = -ones([Mtest N]);

disp('Querying each category')
for c = 1:N
    if(mod(c,5)==0)
        fprintf('.');
    end
    [foo, j] = LMquery(train_imdb, 'object.name', names{c}, 'exact');
    class_training(j,c) = 1;
    [foo, j] = LMquery(test_imdb, 'object.name', names{c}, 'exact');
    class_test(j,c) = 1;
end
fprintf('\n');

% gist Parameters:
clear param
param.imageSize = [256 256];
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

disp('Computing the gist features')
gist_training = LMgist(train_imdb, HOMEIMAGES, param, HOMEGIST);
gist_test = LMgist(test_imdb, HOMEIMAGES, param,HOMEGIST_TEST);
fprintf('\n');
% training
% svm parameters
lambda = 1;

% building kernel
% Compute the kernel:
global K
normD = sum(gist_training.^2,2);
dist = (repmat(normD ,1,size(gist_training,1)) + ...
        repmat(normD',size(gist_training,1),1) - ...
        2*gist_training*gist_training');
K = exp(-0.5/.6^2 * dist);
p_b_gist_test = zeros(Mtest, N);
disp('Kernel logistic regression per category')
fprintf('[/%d]',floor(N/10));
for c = 1:N
    if(mod(c,10)==0)
        fprintf('%d..',c/10);
    end
    %Intermediate non-linear svm
    Y = class_training(:,c);
    [beta,bias,~]=svm_nl(Y, lambda);     
    s = K*beta+bias;

    % fit p(b|gist)
    [logitCoef,dev] = glmfit(s, Y==1, 'binomial');
    
    % test  p(b| gist)
    % Test kernel
    norm1 = sum(gist_test.^2,2);
    dist = (repmat(norm1 ,1,size(gist_training,1)) + ...
            repmat(normD',size(gist_test,1),1) - ...
            2*gist_test*gist_training');
    Kt = exp(-.5/.6^2 * dist);
    st = Kt*beta+bias;
    
    p_b_gist_test(:,c) = glmval(logitCoef, st, 'logit');
   
end
clear K Kt dist