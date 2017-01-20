function [X_,Y_] = virtual_data( X,Y,p,angle,sz)
global doviz
	X_=[];
    Y_=[];
    S = 0;
    textprogressbar('Adding virtual data :  ');
    for i=1:length(Y)
        textprogressbar(i,length(Y));
        if rand(1) < p
            im  = reshape(X(i,:),sz); 
            % Rotate
            r=2*rand(1)-1;
            im1 = imrotate(im, r*angle,'bilinear', 'crop');
            if (rand(1)>.98) && doviz
                figure(1);clf;
                vl_tightsubplot(2,1,'box','inner') ;
                imshow(im) ;
                text(3,3,'Original',...
               'background','w') ;
                vl_tightsubplot(2,2,'box','inner') ;
                imshow(im1) ;
                text(3,3,sprintf('Rotated %.2f',r*angle),...
               'background','w') ;
                drawnow
            end
            X_(end+1,:)= im(:)';
            X_(end+1,:)= im1(:)';
            Y_(end+1)  = Y(i);
            Y_(end+1)  = Y(i);
            S = S+1;
        else
            X_(end+1,:) = X(i,:);
            Y_(end+1) = Y(i);
        end
    end
    textprogressbar('done');
    Y_= Y_';
    fprintf('\nAdded %d images.\n',S);
