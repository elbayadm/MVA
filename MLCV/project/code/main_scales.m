%% Maximum Likelihood parameter estimation for pairwise terms
%% addpaths
addpath(genpath('.'));
addpath('/usr/local/cellar/vlfeat/toolbox')
addpath(genpath('../mlcv_toolbox'))
addpath('/usr/local/cellar/matconvnet/matlab')
vl_setup
vl_setupnn
clearvars; clc;

%% Offsets, means and stds
disp('Computing dataset statistics...')
for im_id=1:1000,
    if ~mod(im_id,100)
        fprintf('.')
    end
    [input_image,points] = load_im(im_id,1,1); %with normalization
    center = points(:,5);
    offsets(:,:,im_id) = points(:,1:4) - center*ones(1,4);
end
fprintf('\n')

for pt = [1:4],
    mn{pt} = mean(squeeze(offsets(:,pt,:)),2);
    sg{pt} = sqrt(diag(cov(squeeze(offsets(:,pt,:))')));
end

part = 2; 
if part ==1 % Bounding boxes
    scales = [1/2 2^(-.5) 1 2^(.5) 2];
    threshold = -.5;
    for im_id = [3 231 507]
        [input_image, bboxes_nose, bboxes] = get_parts_location(im_id,mn,sg,offsets,scales,threshold,1);
        print(sprintf('figures/part3/scales_heat_%d',im_id),'-dpdf')
        parts = {'left eye','right eye','left mouth','right mouth','nose'};

        figure(2); clf;
        showboxes(input_image,bboxes_nose);
        title('Bounding boxes at root')
        print(sprintf('figures/part3/bbox_nose_%d',im_id),'-dpdf')
        clf;
        % With non-maximum susppression:
        pick = nms(bboxes_nose, .7);
        showboxes(input_image,bboxes_nose(pick,:));
        title('Bounding boxes with nms at root')
        print(sprintf('figures/part3/nms_bbox_nose_%d',im_id),'-dpdf')


        for pt = 1:4
            figure(2+pt); clf;
            showboxes(input_image,bboxes{pt},0);
            title(sprintf('Bounding boxes - %s',parts{pt}))
            print(sprintf('figures/part3/bbox_%s_%d',parts{pt},im_id),'-dpdf')

            % With non-maximum susppression:
            pick = nms(bboxes{pt},.7);
            showboxes(input_image,bboxes{pt}(pick,:),0);
            title(sprintf('Bounding boxes with nms - %s',parts{pt}))
            print(sprintf('figures/part3/nms_bbox_%s_%d',parts{pt},im_id),'-dpdf')

        end
    end
end

if part==2 % Precision-recall curve
    I = 501:600;
    [rec,prec,ap] = MLCV_evaldet(I,ones(length(I),1),mn,sg,offsets);
    figure(1); clf;
    % plot precision/recall
    hold on
    for j=1:length(prec)
        plot(rec{j},prec{j},'-');
    end
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(sprintf('class: nose, #subset: %d, AP = %.3f',length(501:600),ap));
end

