function [] = gridlines(x,freq,col,sty,axis)
%col = [.2,.2,.2,.7];
%sty = ':';
if ~exist('axis','var')
    axis =1;
end
if axis ==1
sub =x(1:freq:length(x));
for tick = sub
    plot([tick tick], ylim,'color',col,'linewidth',.5,'linestyle',sty)
    hold on
end
else
sub =x(1:freq:length(x));
for tick = sub
    plot(xlim, [tick tick],'color',col,'linewidth',.5,'linestyle',sty)
    hold on
end
end
