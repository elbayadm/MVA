function [input_image, bboxes_nose, bboxes] = get_parts_location(im_id, label, mn, sg, offsets,scales,threshold,verb)
    % Get bounding boxes of the nose and the remaining parts using GDT for message passing.
    % Inputs:
    % im_id : input image id
    % label : 1 for face image or 0 for background
    % mn, sg : means and stds statistics
    % Offsets Xp-Xr on the training set
    % range of scales for multi-scale detection
    % threshold for box selection
    % verb : verbosity and visualization
    
    %--------------- computing unary terms
    useSVM = 0;
    if verb
        disp('Computing unary terms...')
    end
    if useSVM
        t= load('svm_linear');
    else
        load('svm_cnn2');
    end
    for part = 1:5
        if useSVM
            weights_unary(part,:) = t.svm_linear{part}.weight;
        else
            weights_unary(part,:) = svm_cnn2{part};
        end
    end

    if ~useSVM
        cnn_net = load('data/models/imagenet-caffe-alex.mat');
        cnn_net = vl_simplenn_tidy(cnn_net);
    end

    candidates = {};
    if verb
    disp('Message passing per scale...')
    end

    v0 = 3;
    for sc = 1:length(scales)
        candidates{sc} = {};
        scale  = scales(sc);
        if verb
        fprintf('Scale %.3f\n',scale)
        end
        [input_image, ~] = load_im(im_id,label,0); %without normalization
        input_image  =  imresize(input_image,size(input_image)*scale,'bilinear');
        if useSVM
            [feats,~,idxs]= get_features(input_image,'SIFT');
        else
            [feats,~,idxs]= get_features(input_image,'CNN2',[],cnn_net);
        end
        responses     = weights_unary*feats;
        [sv,sh]       = size(input_image);
        for pt_ind = [1:5],
            score       = -v0*ones(sv,sh);
            score(idxs) = responses(pt_ind,:);
            score_part{sc}{pt_ind} = score;
        end
        % GDT messaging
        for pt = 1:4
            if verb
            fprintf(' Part %d\n',pt)
            end
            sch = sg{pt}(1);
            scv = sg{pt}(2);
            mh  = -mn{pt}(1);
            mv  = -mn{pt}(2);
            
            def(1) = 1/(2*sch^2);
            def(2) = -2*mh/(2*sch^2);
            def(3) = 1/(2*scv^2);
            def(4) = -2*mv/(2*scv^2);
            
            [messages{sc}{pt},ix{pt},iy{pt}] = dt(squeeze(score_part{sc}{pt}),def(1),def(2),def(3),def(4));
            offset =  mh^2/(2*sch^2) + mv^2/(2*scv^2);
            messages{sc}{pt} = messages{sc}{pt} - offset;
            candidates{sc}{pt} =  {iy{pt}, ix{pt}};
        end
    end

    
    if verb
        figure(1); clf;
    end
    for sc = 1:length(scales)
        scale = scales(sc);
        % Locate the root
        belief_nose{sc} =  squeeze(score_part{sc}{5});
        for pt = [1:4], 
            belief_nose{sc}  = belief_nose{sc} + messages{sc}{pt};
        end
        if verb
            if sc==1
                ax1 = subplot(2,3,1);
                colormap(ax1, gray)
                imagesc(input_image);
                title('input')
                set(gca,'xtick',[],'ytick',[]) ; axis image ;
            end

            ax2 = subplot(2,3,sc+1);
            imagesc(max(belief_nose{sc},-v0));
            set(gca,'xtick',[],'ytick',[]) ; axis image ;
            title(sprintf('scale %.3f',scales(sc)))
            colormap(ax2, hot)
        end
    end

    [input_image,points] = load_im(im_id,1,0); %without normalization
    % Locate the root:
    bboxes_nose = [points(1,5) points(2,5) points(1,5) points(2,5) 1 1];
    for sc = 1:length(scales)
        temp = locate_part(belief_nose{sc},threshold,scales(sc));
        if temp
            bboxes_nose = [bboxes_nose; temp];
        end
    end

    bboxes = {};
    for k = 1:size(bboxes_nose,1)
        if k == 1 % ground truth
            for pt = 1:4
                bboxes{pt} =  [points(1,pt) points(2,pt) points(1,pt) points(2,pt) 1 1];
            end
        else
            sc = find(scales==bboxes_nose(k,6));
            nose_location = round(scales(sc)*bboxes_nose(k,1:4));
            x1 = nose_location(1);  x2 =  nose_location(3);
            y1 = nose_location(2);  y2 =  nose_location(4);
            
            for pt = 1:4      
                % map to the messages:
                P = { candidates{sc}{pt}{1}(y1:y2,x1:x2), candidates{sc}{pt}{2}(y1:y2,x1:x2)};
                P{1} = round(P{1}/scales(sc)) ; P{2} = round(P{2}/scales(sc)); 
                mx = min(P{1}(:)) ; Mx = max(P{1}(:)) ;  my = min(P{2}(:));  My = max(P{2}(:)); 
                loc = [my mx My Mx bboxes_nose(k,5) bboxes_nose(k,6)];
                bboxes{pt} = [bboxes{pt}; loc];
            end
        end
    end
