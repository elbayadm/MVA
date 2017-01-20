%% Maximum Likelihood parameter estimation for pairwise terms
%% addpaths
addpath(genpath('.'));
addpath('/usr/local/cellar/vlfeat/toolbox')
addpath(genpath('../mlcv_toolbox'))
addpath('/usr/local/cellar/matconvnet/matlab')
vl_setup
vl_setupnn

%% Offsets, means and stds
for im_id=1:1000,
    [input_image,points] = load_im(im_id,1,1);
    center = points(:,5);
    offsets(:,:,im_id) = points(:,1:4) - center*ones(1,4);
end

for pt = [1:4],
    mn{pt} = mean(squeeze(offsets(:,pt,:)),2);
    sg{pt} = sqrt(diag(cov(squeeze(offsets(:,pt,:))')));
end

% take a look at the data
strs = {'left eye','right eye','left mouth','right mouth','nose'};

clrs = {'r','g','b','k','y'};
figure,
for pt = [1:4],
    scatter(squeeze(offsets(1,pt,:)),squeeze(offsets(2,pt,:)),clrs{pt},'filled'); hold on,
    text(mn{pt}(1),mn{pt}(2),strs{pt},'fontsize',30)
end
axis ij; axis equal;

%% compute unary terms
useSVM = 0;
if useSVM
    t= load('svm_linear');
else
    load('svm_cnn2');
    cnn_net = load('data/models/imagenet-caffe-alex.mat');
    cnn_net = vl_simplenn_tidy(cnn_net);
end
for part = 1:5
    if useSVM
        weights_unary(part,:) = t.svm_linear{part}.weight;
    else
        weights_unary(part,:) = svm_cnn2{part};
    end
end

im_id         = 507;
[input_image] = load_im(im_id,1,1);
if useSVM
    [feats,~,idxs]= get_features(input_image,'SIFT');
else
    [feats,~,idxs]= get_features(input_image,'CNN2',[],cnn_net);
end
responses     = weights_unary*feats;
[sv,sh]       = size(input_image);
for pt_ind = [1:5],
    score       = -10*ones(sv,sh);
    score(idxs) = responses(pt_ind,:);
    score_part{pt_ind} = score;
end

figure
subplot(2,3,1); imshow(input_image);
for pt_ind = [1:5],
    subplot(2,3,1+pt_ind);
    imshow(score_part{pt_ind},I);
    if useSVM
        title([strs{pt_ind},' with SVM-Linear - SIFT']);
    else
        title([strs{pt_ind},' with SVM-Linear - CNN2']);
    end
end

% dt- potential:  def(1) h^2 + def(2) h + def(3) * v^2 + def(4) *v
% gaussian potential:   (h - mh)^2/(2*sch^2) + (v-mv)^2/(2*scv^2)
% GDT messaging
for pt = [1:4]
    sch = sg{pt}(1);
    scv = sg{pt}(2);
    mh  = -mn{pt}(1);
    mv  = -mn{pt}(2);
    
    def(1) = 1/(2*sch^2);
    def(2) = -2*mh/(2*sch^2);
    def(3) = 1/(2*scv^2);
    def(4) = -2*mv/(2*scv^2);
    
    [mess_gdt{pt},ix{pt},iy{pt}] = dt(squeeze(score_part{pt}),def(1),def(2),def(3),def(4));
    offset =  mh^2/(2*sch^2) + mv^2/(2*scv^2);
    mess_gdt{pt} = mess_gdt{pt} - offset;
end

% Max-product message passing:
mess_maxp  = {};
fprintf('Messages from parts to root')
for pt = [1:4]
    fprintf('\nPart %d ',pt)
    mess_maxp{pt} = zeros(sv,sh);
    sch = sg{pt}(1);
    scv = sg{pt}(2);
    mh  = mn{pt}(1);
    mv  = mn{pt}(2);
    
    def(1) = -1/(2*sch^2);
    def(2) = mh/(sch^2);
    def(3) = -1/(2*scv^2);
    def(4) = mv/(scv^2);
    for xr=1:(sv*sh)
        if ~mod(xr,2000)
            fprintf('.')
        end
        [xr1, xr2] = ind2sub([sv, sh],xr);      
        psi = def(1)*repmat(((1:sh)-xr2).^2,sv,1) +def(2)*repmat((1:sh)-xr2,sv,1)+...
              def(3)*repmat(((1:sv)'-xr1).^2,1,sh)+def(4)*repmat((1:sv)'-xr1,1,sh);
        mess_maxp{pt}(xr1,xr2) = max(score_part{pt}(:)+psi(:));
    end
  offset =  mh^2/(2*sch^2) + mv^2/(2*scv^2);  
  mess_maxp{pt} = mess_maxp{pt} - offset;
end

% Root to leaves:
mess_maxp_root  = {};
mess_maxp_rest = {};
fprintf('\nMessages from root to parts')
for pt = [1:4]
    fprintf('\nPart %d:\n',pt)
    % Messages from other neighbors
    mess_maxp_rest{pt} = zeros(sv,sh);
    for k=setdiff(1:4,pt)
      mess_maxp_rest{pt} = mess_maxp_rest{pt} + mess_maxp{k};
    end
    mess_maxp_root{pt} = zeros(sv,sh);
    sch = sg{pt}(1);
    scv = sg{pt}(2);
    mh  = mn{pt}(1);
    mv  = mn{pt}(2);
    
    def(1) = -1/(2*sch^2);
    def(2) = mh/(sch^2);
    def(3) = -1/(2*scv^2);
    def(4) = mv/(scv^2);
    for xp=1:(sv*sh)
        if ~mod(xp,2000)
            fprintf('.')
        end
        [xp1, xp2] = ind2sub([sv, sh],xp);      
        psi = def(1)*repmat(((1:sh)-xp2).^2,sv,1) -def(2)*repmat((1:sh)-xp2,sv,1)+...
              def(3)*repmat(((1:sv)'-xp1).^2,1,sh)-def(4)*repmat((1:sv)'-xp1,1,sh);
        mess_maxp_root{pt}(xp1,xp2) = max(score_part{5}(:)+psi(:)+ mess_maxp_rest{pt}(:));
    end
  offset =  mh^2/(2*sch^2) + mv^2/(2*scv^2);  
  mess_maxp_root{pt} = mess_maxp_root{pt} - offset;
end
%% Plot the messages:
belief_nose_gdt  =  squeeze(score_part{5});
belief_nose_maxp =  squeeze(score_part{5});
I = [-1 1];
v0 = 1;
%I = [-2 2];
%v0 = 10;
parts = {'left eye','right eye','left mouth','right mouth','nose'};
for pt = [1:4],
    figure(pt); clf;
    vl_tightsubplot(1,3,1,'box','outer');
    imshow(squeeze(score_part{pt}),I); title(['\phi_{',parts{pt},'}(X)'],'fontsize',10);
    vl_tightsubplot(1,3,2,'box','outer');
    imshow(mess_gdt{pt},I); title(['m_{',parts{pt},'-> nose}(X) - GDT'],'fontsize',10);
    vl_tightsubplot(1,3,3,'box','outer');
    imagesc(mess_maxp{pt},I); title(['m_{',parts{pt},'-> nose}(X) - Max-product'],'fontsize',10);
    colormap hot
    set(gca,'xtick',[],'ytick',[]) ; axis image ;
    belief_nose_gdt  = belief_nose_gdt + mess_gdt{pt};
    belief_nose_maxp = belief_nose_maxp + mess_maxp{pt};
    print(sprintf('figures/mess_pt%d_%d',pt,im_id),'-dpdf')
end

figure(5),
ax1 = subplot(1,3,1);
colormap(ax1, gray)
imagesc(input_image);
title('input')
set(gca,'xtick',[],'ytick',[]) ; axis image ;

ax2 = subplot(1,3,2);
imagesc(max(belief_nose_gdt,-v0));
set(gca,'xtick',[],'ytick',[]) ; axis image ;
title('GDT')
colormap(ax2, hot)

ax3=subplot(1,3,3);
imagesc(max(belief_nose_maxp,-v0));
set(gca,'xtick',[],'ytick',[]) ; axis image ;
title('Max-product')
colormap(ax3, hot)
print(sprintf('figures/belief_nose_%d',im_id),'-dpdf')

% Plot the leaves messages:
for pt = 1:4,
    figure(5+pt);clf;
    imagesc(mess_maxp_root{pt},I);
    colormap hot
    set(gca,'xtick',[],'ytick',[]) ; axis image ;
    title(['m_{nose ->',parts{pt},'}(X)']);
    print(sprintf('figures/message_%s_%d',parts{pt},im_id),'-dpdf')
end
%% show ground-truth bounding box. 
%% You will need to adapt this code to make it show your bounding box proposals
[input_image,points] = load_im(im_id,1,1);
figure(10);
min_x = min(points(1,:));
max_x = max(points(1,:));
min_y = min(points(2,:));
max_y = max(points(2,:));
score = 1;
bbox  = [min_x,min_y,max_x,max_y,score];
showboxes(input_image,bbox);
