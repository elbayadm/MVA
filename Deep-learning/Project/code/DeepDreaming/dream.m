function [dreams]= dream(net, im, opts)
global isDag
layer=opts.layer;
iters=opts.iters;
% For deep-dreaming on multiple layers
if length(layer)>1 && ~isfield(opts,'combine')
   opts.combine=ones(1,length(layer))/length(layer);
end
%% Parameters for printing and saving the output files:
    if isDag
        if length(layer)>1
            lindex='';
            lname='';
            for i=1:length(layer)
                l=layer(i);
                lname=[lname '+' net.layers(l).name];
                lindex=[lindex '+' num2str(opts.combine(i)) 'x' int2str(l)];
            end
            lname=lname(2:end);
            lindex=lindex(2:end);
        else
            lname=net.layers(layer).name;
            lindex=int2str(layer);
        end

    elseif isfield(net.layers{layer(end)}, 'name')
        if length(layer)>1
            lindex='';
            lname='';
            for i=1:length(layer)
                l=layer(i);
                lname=[lname '+' net.layers{l}.name];
                lindex=[lindex '+' num2str(opts.combine(i)) 'x' int2str(l)];
            end
            lname=lname(2:end);
            lindex=lindex(2:end);
        else
            lname=net.layers{layer}.name;
            lindex=int2str(layer);
        end
    else
        lname=''; lindex='';
    end
    opts.lindex=lindex;
    opts.lname=lname;
%% Main loop ------------------------------------------------------------------||
fprintf('Layer %s (%s): ',lindex,lname);
dreams=[];
dreams.objective=[];
dreams.octaves={};
dreams.crops={};
% Pre-processing the image:
im_ = process_im(im);
switch opts.crop
case 0 %straightforward dreaming
    nc=1;
    [~, dreams] =iterative_zooming(net,{im_},opts,dreams);
    if opts.doprint
        if ~exist(['figures/' opts.fig],'dir')
            mkdir('figures/', opts.fig);
        end
        if ~exist([opts.fig, '/', opts.objective],'dir')
           mkdir(['figures/' opts.fig '/'], opts.objective);
        end
        print('-f2',sprintf('figures/%s/%s/L%s_oct%d_step%d_iters%d_%s',...
            opts.fig,opts.objective,opts.lindex,opts.octave+1,floor(opts.step*10),iters,opts.net), '-dpdf')
    end
case 1 %cropping along a grid
    im__=im_;
    Ws=floor(linspace(1,size(im_,1),opts.grid+1));
    Hs=floor(linspace(1,size(im_,2),opts.grid+1));
    [gridW,gridH]=meshgrid(1:opts.grid,1:opts.grid);
    nc=((opts.grid)^2);
    for c=1:nc
        rmin=max(Ws(gridW(c))-opts.strip,1);
        rmax=min(Ws(gridW(c)+1)+opts.strip,size(im_,1));
        cmin=max(Hs(gridH(c))-opts.strip,1);
        cmax=min(Hs(gridH(c)+1)+opts.strip,size(im_,2));
        crop=im__(rmin:rmax,cmin:cmax,:,:);
        crop=imresize(crop,opts.scale);
        fprintf('\nCrop %d - %dx%d (%dx%dx%d)\n',c,gridW(c),gridH(c),size(crop))
        [octaves, dreams] =iterative_zooming(net,{crop},opts,dreams);
        crop=octaves{1}-crop;
        crop=imresize(crop,[rmax-rmin+1, cmax-cmin+1]);
        crop(1:opts.strip,:,:)=crop(1:opts.strip,:,:)/2;
        crop(end-opts.strip:end,:,:)=crop(end-opts.strip:end,:,:)/2;
        crop(:,1:opts.strip,:)= crop(:,1:opts.strip,:)/2;
        crop(:,end-opts.strip:end,:)=crop(:,end-opts.strip:end,:)/2; 

        im_(rmin:rmax,cmin:cmax,:,:) = im_(rmin:rmax,cmin:cmax,:,:)+ crop;
        % deprocess for outputs:
        dreams.crops{end+1}=deprocess_im(im_);
        if opts.doviz
            figure(1),clf;
            imshow(dreams.crops{end}) ;
            title(sprintf('L%s - step %.1f - crop  %d',lindex,opts.step,c),'interpreter','none','fontsize',10)
            drawnow
            if opts.doprint && c==nc
                if ~exist(['figures/' opts.fig],'dir')
                    mkdir('figures/', opts.fig);
                end
                if ~exist([opts.fig, '/', opts.objective],'dir')
                   mkdir(['figures/' opts.fig '/'], opts.objective);
                end
                print(sprintf('figures/%s/%s/L%s_crop_grid%d_oct%d_step%d_iters%d_%s',...
                    opts.fig,opts.objective,lindex,opts.grid,opts.octave+1,floor(opts.step*10),iters,opts.net), '-dpdf')
            end
        end
    end
otherwise
    im__=im_;
    nc=opts.crop;
    for c=1:nc
        mW=floor(size(im_,1)/2);
        mH=floor(size(im_,2)/2);
        rmin=max(randi([1,size(im_,1)-20],1),1);
        rmax=min(rmin+randi([min(30,mW),mW],1),size(im_,1));
        cmin=max(randi([1,size(im_,2)-20],1),1);
        cmax=min(cmin+randi([min(30,mH),mH],1),size(im_,2));

        crop=im__(rmin:rmax,cmin:cmax,:);
        crop=imresize(crop,opts.scale);
        fprintf('\nCrop %d - %d:%dx%d:%d\n',c,rmin,rmax,cmin,cmax)
        [octaves, dreams] =iterative_zooming(net,{crop},opts,dreams);
        crop=octaves{1}-crop;
        crop=imresize(crop,[rmax-rmin+1, cmax-cmin+1]); 
        im_(rmin:rmax,cmin:cmax,:) = im_(rmin:rmax,cmin:cmax,:)+ crop;
        % deprocess for outputs:
        dreams.crops{end+1}=deprocess_im(im_);
        if opts.doviz
            figure(1),clf;
            imshow(dreams.crops{end}) ;
            title(sprintf('L%s - step %.1f - crop  %d',lindex,opts.step,c),'interpreter','none','fontsize',10)
            drawnow
            if opts.doprint && c==nc
                if ~exist(['figures/' opts.fig],'dir')
                    mkdir('figures/', opts.fig);
                end
                if ~exist([opts.fig, '/', opts.objective],'dir')
                  mkdir(['figures/' opts.fig '/'], opts.objective);
                end
                print(sprintf('figures/%s/%s/L%s_crop_random%d_oct%d_step%d_iters%d_%s',...
                    opts.fig,opts.objective,lindex,nc,opts.octave+1,floor(opts.step*10),iters,opts.net), '-dpdf')
            end
        end
    end
end

%% Visualization
if opts.doviz
figure(3),
set(3,'name', 'objective function', 'units','normalized','position',[.1 .1 .5 .3],...
    'PaperType','A4','PaperPositionMode','auto');
plot(dreams.objective,'color','b','linewidth',2)
if opts.octave
    for o=1:(opts.octave+1)*nc
        line([o*iters o*iters],ylim,...
            'linestyle','-','color',[.8 .8 .8])
    end     
end
str=sprintf('figures/%s/%s/Optim_oct%d_L%s_step%d_iters%d_%s',...
    opts.fig,opts.objective,opts.octave+1,lindex,floor(opts.step*10),iters,opts.net);
if opts.crop
    for c=1:nc-1
        line([c*(opts.octave+1)*iters c*(opts.octave+1)*iters],ylim,...
            'linestyle','--','color','k')
    end
    if opts.crop==1
     str=sprintf('figures/%s/%s/Optim_crop_grid%d_L%s_step%d_iters%d_%s',...
         opts.fig,opts.objective,opts.grid,lindex,floor(opts.step*10),iters,opts.net);
    else
     str=sprintf('figures/%s/%s/Optim_crop_random%d_L%s_step%d_iters%d_%s',...
         opts.fig,opts.objective,opts.crop,lindex,floor(opts.step*10),iters,opts.net);
    end
end

title(['Objective optimization - ' opts.objective ]);
xlabel('iteration')
xlim([1 length(dreams.objective)])
ylabel(sprintf('%s (%s)',lname,lindex),'interpreter','none')
if opts.doprint
    print(str, '-dpdf')
end
end
fprintf('\nDone\n')
end

%% Iterative zooming:
function [octaves, dreams] =iterative_zooming(net,octaves,opts,dreams)
global isDag
for i=1:opts.octave
   octaves{end+1}=imresize(octaves{end},1/opts.scale);
end
dream=zeros(size(octaves{end}),'like',octaves{end});
for o=length(octaves):-1:1
    fprintf('\nOctave %d - %dx%dx%d\n',length(octaves)-o+1,size(octaves{o}))
    if o < length(octaves)
        dream=imresize(dream,[size(octaves{o},1),size(octaves{o},2)]);
    end
    input=octaves{o}+ dream; 
    if strcmp(opts.objective,'guide')
        opts.g_features = guiding_features(input,net,opts);
    end
    if isDag
        net.reset();
    end
    for it =1:opts.iters
        [input, obj]= nn_step(net,input,opts);
        dreams.objective(end+1)=obj;
        %blurring
        if opts.blur && ~mod(it,opts.freq)
            disp('Blurring')
           input=imgaussfilt(input,opts.blur);
        end

        %Averaging:
        if opts.average && ~mod(it,opts.freq)
            disp('Averaging')
            h=fspecial('average',opts.average);
            input = imfilter(input,h,'replicate');
        end
    end
    %Retrieve the dream part:
    dream=input-octaves{o};
    octaves{o}=input;
    % deprocess for outputs:
    dreams.octaves{end+1}=deprocess_im(input);
    if opts.doviz || (o==1)
        figure(2), clf;
        imshow(dreams.octaves{end}) ;
        title(sprintf('L%s - step %.1f - octave %d',opts.lindex,opts.step,length(octaves)-o+1),'interpreter','none','fontsize',10)
        drawnow
    end
end
end
%% Processing the image:
function im_ = process_im(im)
    global sketch
    im_=255*im;
    %im_=im_(:, :, [3, 2, 1]); %swap channels
    %im_ = permute(im_, [2, 1, 3]);  % flip width and height
    global mean_image;
    im_ = bsxfun(@minus, im_, mean_image);
    if sketch 
        im_=cat(4,im_(:,:,1),im_(:,:,2),im_(:,:,3));
        im_=255-im_;
    end
end
%% De-Processing the image:
function im_ = deprocess_im(im)
    global sketch
    im_=im;
    if sketch
        im_=255-im_;
        im_=cat(3,im_(:,:,:,1),im_(:,:,:,2),im_(:,:,:,3));
    end
    global mean_image; 
    im_ = bsxfun(@plus, im_, mean_image);
    %im_ = permute(im_, [2, 1, 3]);  % flip width and height if needed (caffe)
    %im_ = im_(:, :, [3, 2, 1]); %swap channels if needed (caffe)
    im_=im_/255;
    end
%% Compute guiding features:
function g_features = guiding_features(input,net,opts)
    global isDag
    layer=opts.layer;
    % Guiding image
    guide_=imresize(opts.guide,[size(input,1) size(input,2)]);
    guide_= process_im(guide_);
    %compute its activation:
    if isDag
        net.reset();
        % Forward:
        forwardto_dag(net,guide_,layer);
        g_features=net.vars(net.layers(layer).outputIndexes).value;
        g_features=reshape(g_features,[],size(g_features,3));
    else
        resg = init_res(layer) ; 
        resg(1).x = guide_;
        resg = forwardto_nn(net, layer, resg) ;
        %reshape the features:
        g_features=reshape(resg(layer+1).x,[],size(resg(layer+1).x,3));
    end
end
%% Forward & backward step
function [input, obj] = nn_step(net,input,opts)
    layer=opts.layer;
    global isDag
    global debug
    global mean_image
    step_size=opts.step;   
    %jitter
    if opts.jitter
        roll=randi([-opts.jitter,opts.jitter+1],1,2);
        input=circshift(input,roll(1),1);
        input=circshift(input,roll(2),2);
    end
    if isDag
        % Forward:
        forwardto_dag(net,input,max(layer));
        % Backprop:

        switch opts.objective
            case 'N2'  
                dzdx=2*net.vars(net.layers(layer).outputIndexes).value;
                obj=norm(dzdx(:)/2);
            case 'neuron'
                 dzdx=2*net.vars(net.layers(layer).outputIndexes).value; 
                 W=zeros(size(dzdx));
                 W(opts.activate)=1;
                 dzdx=dzdx.*W;
                 obj=norm(dzdx(:)/2);
             case 'sumN2'
                 obj=0;
                 dzdx={};

                 for i=1:length(layer) % opts.layer is an array of layers indices
                     l=layer(i);
                     obj=obj+norm(net.vars(net.layers(l).outputIndexes).value(:));
                     dzdx{end+1}=opts.combine(i)*2*net.vars(net.layers(l).outputIndexes).value;
                 end
              case 'guide'
                  activation_=net.vars(net.layers(layer).outputIndexes).value;
                  activation=reshape(activation_,[],size(activation_,3));
                  dott=activation*opts.g_features';
                  [~,W]=max(dott);
                  dzdx=opts.g_features(W,:);
                  dzdx=reshape(dzdx,size(activation_));
                  obj=norm(dott(:));
        end

        backwardfrom_dag(net, dzdx, layer);

        % Update:
        % the gradient
        g=net.vars(1).der;

    else %SimpleNN

        res = init_res(max(layer)) ; 
        % Forward:
        res(1).x = input;
        res = forwardto_nn(net, max(layer), res) ;
        % Backprop:
        switch opts.objective
            case 'N2'  
                dzdx=2*res(layer+1).x;
                obj=norm(res(layer+1).x(:));
            case 'neuron'
                 dzdx=res(layer+1).x;
                 W=zeros(size(dzdx));
                 W(opts.activate)=1;
                 dzdx=dzdx.*W;
                 %activation=dzdx;
                 obj=norm(dzdx(:));
             case 'sumN2'
                 obj=0;
                 dzdx={};

                 for i=1:length(layer) % opts.layer is an array of layers indices
                     l=layer(i);
                     obj=obj+norm(res(l+1).x(:));
                     dzdx{end+1}=opts.combine(i)*2*res(l+1).x;
                 end
            case 'guide' 
                activation=reshape(res(layer+1).x,[],size(res(layer+1).x,3));
                dott=activation*opts.g_features';
                [~,W]=max(dott);
                dzdx=opts.g_features(W,:);
                dzdx=reshape(dzdx,size(res(layer+1).x));
                obj=norm(dott(:));
        end

        if iscell(dzdx)
            res = backwardfrom_sum_nn(net, layer, dzdx, res);
        else
            res = backwardfrom_nn(net, layer, dzdx, res);
        end   
        % the gradient
        g=res(1).dzdx;
    end

    fprintf(' %.2e..',obj);
    norm2=mean(abs(g(:)));
    g=g/(norm2+1*(norm2==0));

    if opts.clip
        %W=abs(sum(input.*g,3));
        W=sum(input.*input,3);
        tol=opts.clip*mean(W(:));
        if debug 
            fprintf('tolerance %.2e\n',tol);
            fprintf('clipped %d pixels\n',sum(W(:)<tol));
        end
        input=input.*(repmat(W,1,1,3)>tol);
    end
    % Update:
    input=input+step_size*g;
    % Regularization:
    if opts.decay
        input=(1-opts.decay)*input;
    end
    % unjitter
    if opts.jitter
        input=circshift(input,-roll(1),1);
        input=circshift(input,-roll(2),2);
    end
    %clip between -mean and 255-mean
    for i=1:size(input,3)
    input(:,:,i)=-mean_image(i)*(input(:,:,i)<-mean_image(i))+...
        input(:,:,i).*(input(:,:,i)>=-mean_image(i)).*(input(:,:,i)<=(255-mean_image(i)))+...
        (255-mean_image(i))*(input(:,:,i)> (255-mean_image(i)));
    end
end