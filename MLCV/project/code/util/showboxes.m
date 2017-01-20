function showboxes(im, boxes, addscore)
% Draw boxes on top of image.
if nargin <3
  addscore = 1;
end
imagesc(im); 
set(gca,'xtick',[],'ytick',[]) ; axis image ;
colormap gray
% axis equal;
% axis on;
if ~isempty(boxes)
  numboxes = size(boxes, 1);
  for i = 1:numboxes
    x1 = boxes(i,1);
    y1 = boxes(i,2);
    x2 = boxes(i,3);
    y2 = boxes(i,4);
    score = boxes(i,5);
    hold on,
    if i==1 % ground truth
      assert(x1==x2 & y1==y2)
      plot(x1,y1,'xr','linewidth',10)
    else
      plot([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]', 'linewidth', 3);
      if addscore
        text(x1,y1,sprintf('%.1f', score),...
         'color','k',...
         'background','w',...
         'Margin',.1,...
         'verticalalignment','top', ...
         'fontsize', 9) ;
      end
    end 
  end
end
drawnow;
