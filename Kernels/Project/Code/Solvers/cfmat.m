function [cm]=cfmat(pred,gt,labels)
% Confusion matrix
	c = length(labels);
	cm = zeros(c);
    for ci=1:c
		I = find(gt==labels(ci));
	    for cj=1:c
	        cm(ci,cj) = sum(pred(I)==labels(cj))/length(I);
	    end
    end
    draw_cm(cm,labels,c);

end

function draw_cm(cm,tick,c)
matshow = log(cm);
imagesc(1:c,1:c,matshow);

legends = num2str(cm(:),'%0.3f');
legends = strtrim(cellstr(legends));

for i = 1:length(legends)
    if isequal(legends(i),{'0.000'})
        legends(i) = {''};
    end
end

[x,y] = meshgrid(1:c); 
text(x(:),y(:),legends(:), 'HorizontalAlignment','center');
xlabel('prediction')
ylabel('ground truth')
set(gca,'xticklabel',tick,'XAxisLocation','top');
set(gca, 'XTick', 1:c, 'YTick', 1:c);
set(gca,'yticklabel',tick);
end