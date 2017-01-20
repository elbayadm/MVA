function [rec,prec,ap] = MLCV_evaldet(image_ids,labels,mn,sg,offsets)

% ground truth objects.
gt =[]; BB =[]; confidence=[]; ids = [];
npos=sum(labels);

% Gathering gt locations and bounding boxes
for i = 1:length(image_ids)
    if ~mod(i,10)
        fprintf('.')
    end
    im_id = image_ids(i);
    th = -3;
    [input_image, bboxes_nose, bboxes] = get_parts_location(im_id,labels(i),mn,sg,offsets,[.5 1 1.4 2],th,0);
    center = bboxes_nose(1,:);
    gt(i).BB = [center(1)-20 center(2)-20 center(1)+20 center(2)+20];

    % With non-maximum susppression:
    pick = nms(bboxes_nose, .7);
    
    bboxes_nose = bboxes_nose(pick,:);
    boxes = bboxes_nose(2:end,1:4)'; scores = bboxes_nose(2:end,5)';
    BB =  [BB,boxes];
    confidence  = [confidence,scores];
    ids = [ids,i*ones(size(scores))];
    gt(i).det = false;
end

% at the end: 
% BB should be a 4 x N array containing all N bounding boxes
% (4 entries are x1-left,y1-top,x2-right,y2-bottom - and y1<y2!!)
% confidence should be a 1 x N vector with the respective scores 

% sort detections by decreasing confidence
[sc,si]=sort(-confidence);
% try different threshold for box proposal:
ths = [-3 -1 -0.5 .5 1];
rec={}; prec={}; ap={};
for it  = 1:length(ths)
    st = find(sc==ths(it),1,'last');
    ids_=ids(si(st+1:end));
    BB_=BB(:,si(st+1:end));
    confidence_ = sc(st:end)
    % assign detections to ground truth objects
    nd=length(confidence_);
    tp=zeros(nd,1);
    fp=zeros(nd,1);
    for d=1:nd
        % find ground truth image
        i=ids_(d);
        % ground truth bounding box
        bbgt=gt(i).BB;
        % proposed box
        bb=BB_(:,d);
        
        ov = boxoverlap(bb',bbgt);
        %fprintf('overlap %.3f\n',ov)
        % assign detection as true positive/don't care/false positive
        ov_threshold = .4;
        if ov>=ov_threshold
            if ~gt(i).det
                tp(d)=1;            % true positive
                gt(i).det = true;
            else
                fp(d)=1;            % false positive (multiple detection)
            end
        else
            fp(d)=1;                    % false positive
        end
    end
    % compute precision/recall
    fp=cumsum(fp);
    tp=cumsum(tp);
    rec{end+1}=tp/npos;
    prec{end+1}=tp./(fp+tp);
    ap{end+1}=VOCap(rec{end},prec{end});
end
