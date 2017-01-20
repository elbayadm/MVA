%% addpaths
addpath(genpath('.'));
addpath('/usr/local/cellar/vlfeat/toolbox')
addpath(genpath('../mlcv_toolbox'))
addpath('/usr/local/cellar/matconvnet/matlab')
vl_setup
vl_setupnn
%% take a look into the problem
im_id = 1;
label = 1;
[im,pts] = load_im(im_id,label);
figure(1);clf;
imshow(im);
hold on,scatter(pts(1,:),pts(2,:),'g','filled');

label  = 0;
[im,pts] = load_im(im_id,label);
figure(2); clf;
imshow(im);
hold on,scatter(pts(1,:),pts(2,:),'r','filled');

%-------------------------------
%% Dataset: train-val-test splits
%--------------------------------
% images with faces
total_positives = 30;      %% set to 1500 by the end
train_positives = [1:2:total_positives];
test_positives  = [2:2:total_positives];

% background images (seem few, but we draw multiple negatives samples per image)
total_negatives = 10;       %% set to 200 by the end
train_negatives = [1:2:total_negatives];
test_negatives  = [2:2:total_negatives];


train_set = [train_positives,            train_negatives];
train_lbl = [ones(size(train_positives)),zeros(1,length(train_negatives))];

test_set  = [test_positives,             test_negatives];
test_lbl  = [ones(size(test_positives)), zeros(1,length(test_negatives))];

%-------------------------------
%% Experiment setup
%--------------------------------
doprint = 0; downsize = 0;
%feature_names       = {'SIFT','CNN2','CNN5'};
feature_names       = {'CNN2'};
part_names          = {'Left eye','Right eye','Left mouth','Right mouth','Nose'};
%classifier_names    = {'Linear','Logistic','SVM','SVM-RBF'};
classifier_names    = {'SVM'};
sigmoid = @(x) 1./(1+exp(-x));

svm_cnn2 = {} ; % structure to store the model
for feature_ind = 1:length(feature_names)
    feat = feature_names{feature_ind};
    for part = 1:length(part_names)
        part_name = part_names{part};

        % Step 1: gather training dataset ----------------------------------------------------------------------------
        normalize = 1;  % makes sure faces come at a fixed scale 
        features  = [];
        labels    = [];
        fprintf('Gathering training set: \n0+ ');

        % Load the reference CNN model and keep only the necessary layers to avoid
        % unnecessary computations.
        %cnn_net = dagnn.DagNN.loadobj(load('data/models/imagenet-matconvnet-alex.mat'));
        cnn_net = load('data/models/imagenet-caffe-alex.mat');
        cnn_net = vl_simplenn_tidy(cnn_net);
        vl_simplenn_display(cnn_net)
        for im_idx = 1:length(train_set)
            image_id  = train_set(im_idx);
            image_lb  = train_lbl(im_idx);
            
            [input_image,points]    = load_im(image_id,image_lb,normalize,part);
            features_im             = get_features(input_image,feat,points,cnn_net);
            reject = any((isnan(features_im)|isinf(features_im)),1); features_im(:,reject) = [];

            features                = [features,features_im];
            labels                  = [labels,  image_lb*ones(1,size(features_im,2))];
            
            fprintf(2,' %i',mod(im_idx-1,10));
            if mod(im_idx,10)==0, fprintf(2,'\n%i+ ',im_idx); end
        end
        if downsize
            features = features(:,1:300);
            labels = labels(:,1:300);
        end
        fprintf('\n')
        % Step 2: gather test dataset ----------------------------------------------------------------------------
        test_features  = [];
        test_labels    = [];
        fprintf('Gathering test set: \n0+');

        for im_idx = 1:length(test_set)
            image_id  = test_set(im_idx);
            image_lb  = test_lbl(im_idx);
            
            [input_image,points]    = load_im(image_id,image_lb,normalize,part);
            features_im             = get_features(input_image,feat,points,cnn_net);
            
            reject = any((isnan(features_im)|isinf(features_im)),1); features_im(:,reject) = [];   
            test_features                = [test_features,features_im];
            
            test_labels                  = [test_labels,  image_lb*ones(1,size(points,2))];
            
            fprintf(2,'.%i',mod(im_idx-1,10));
            if mod(im_idx,10)==0, fprintf(2,'\n%i+ ',floor(im_idx/10)); end
        end
        if downsize
            test_features = test_features(:,1:50);
            test_labels = test_labels(:,1:50);
        end
        fprintf('\n')
        %----------------------------------------------------------------------------
        legends={};
        for ind  =  1:length(classifier_names)
            classifier_name  =  classifier_names{ind};
            experiment_string = sprintf('Features_%s_Part_%s_Classifier_%s',feat,part_name,classifier_name);
            disp(experiment_string)
            % Step 3: train classifier 
            switch lower(classifier_name)
                case 'linear'    %% I can do this 
                    w = (labels*features')*pinv(features*features');
                    test_scores = w*test_features;
                case 'logistic'  
                    w = logit_cv_newton(features,labels);
                    print(['figures/part1/' experiment_string '_logiterror'],'-dpdf')
                    test_scores = sigmoid(w*test_features);
                case 'svm'
                    w = train_cv_linear_svm(features,labels);
                    print(['figures/part1/' experiment_string '_linerror'],'-dpdf')
                    test_scores = w*test_features;
                case 'svm-rbf'
                    w = train_cv_rbf_svm(features,labels);
                    print(['figures/part1/' experiment_string '_rbferror'],'-dpdf')
                    test_scores = w*test_features;
            end
            svm_cnn2{part} = w;

            % Step 4: Precision recall curve for classifier
            
            [recall, precision,info] = vl_pr(2*test_labels-1, test_scores);
            
            if ind==1
                figure(1); clf; hold on,
                rp = sum(labels==1) /length(labels) ;
                plot([0 1], [rp rp],'--k','linewidth', 2) ;
                legends{end+1} = 'random';
            end
            figure(1); hold on,
            plot(recall,precision,'linewidth',2) ;
            legends{end+1} = sprintf('%s - AP %.2f%%',classifier_name,info.ap*100);
            
                        
            % Step 5: fun code: see what the classifier wants to see
            if strcmp('SIFT',feat)   
                figure(2);clf;
                vl_plotsiftdescriptor(max(w(1:end-1)',0)); 
                title('positive components of weight vector');
                if doprint
                    print(['figures/part1/' experiment_string '_sift_pos'], '-dpdf')
                end
                figure(2);clf;
                vl_plotsiftdescriptor(max(-w(1:end-1)',0));
                title('negative  components of weight vector');
                if doprint
                    print(['figures/part1/' experiment_string '_sift_neg'],'-dpdf')
                end

            end

            % Step 6: Dense evaluation of classifier    
            for image_lb = [0,1] 
                % try both a positive and a negative image
                [input_image,points]       = load_im(image_id,image_lb,normalize,part); 
                % important: make sure you do NOT give the third, 'points', argument
                [dense_features,crds,idxs] = get_features(input_image,feat,[],cnn_net);
                % dense_features will be Nfeats X Npoints. 
                % output porbailities instead of score
                %score_classifier = sigmoid(w*dense_features); %TODO: use platt's scaling
                score_classifier = w*dense_features;
                
                [sv,sh]     = size(input_image);
                score       = zeros(sv,sh);
                score(idxs) = score_classifier;
                
                title_string    = sprintf('%s score for part: %s\n Feature: %s',classifier_name,part_name,feat);
                figure(4+image_lb); clf; 
                %vl_tightsubplot(1+image_lb,2-image_lb,1,'box','outer');
                imagesc(score); set(gca,'xtick',[],'ytick',[]) ; axis image ;
                colormap gray
                %title(title_string);
                %vl_tightsubplot(1+image_lb,2-image_lb,2,'box','outer');
                %imagesc(input_image); set(gca,'xtick',[],'ytick',[]) ; axis image ;
                %colormap gray
                if doprint
                    print(['figures/part1/' experiment_string '_dense_' num2str(image_lb)],'-dpdf')
                end
            end

        end
        if doprint
            figure(1);
            title_string    = ...
                sprintf('PR curve - part: %s - feature: %s',part_name,feat);
            
            legend(legends,'location','southwest')
            axis square ; grid on ;
            xlim([0 1]) ; xlabel('recall') ;
            ylim([0 1]) ; ylabel('precision') ;
            title(title_string);
           print('-f1',['figures/part1/' sprintf('Features_%s_Part_%s',feat,part_name) '_pr'],'-dpdf')
        end
    end            
end
