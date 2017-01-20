function [] = plot_hist(input,output,s_list,s_modes,fp, Dp,varargin)
col = [9 125 18;199 2 45;145 145 145;252 215 5]/255;
if nargin == 7
    antimodes = varargin{1};
end
%-----------------------------------------------------------------
subplot(1,2,1)
plot(s_list,fp,'color','b','linewidth',2)
hold on,
%line([input input],ylim,'color','k','linewidth',2)
line([output output],ylim,'color','g','linewidth',3,'linestyle','--')
hold on,
gridlines(s_list,1,[.2,.2,.2,.7],':')
hold on,
gridlines(s_modes',1,col(1,:),'-')
if exist('antimodes','var')
gridlines(s_list(antimodes),1,col(2,:),'-')
c=4;
for i=1:length(antimodes)-1
    h = area(s_list(antimodes(i):antimodes(i+1)),...
        fp(antimodes(i):antimodes(i+1)));
    set(h,'facecolor',col(c,:))
    alpha(.5)
    if (c == 4), c=3; else c=4; end
end
end
xlabel('intensity')
title('Frequency')
axis 'tight'
axis 'square'
set(gca,'ytick',[])
%-----------------------------------------------------------------
subplot(1,2,2)
plot(s_list,Dp,'color','b','linewidth',2)
hold on,
line([0 1], [0 0],'color','k')
hold on,
gridlines(s_list,1,[.2,.2,.2,.7],':')
hold on,
gridlines(s_modes',1,col(1,:),'-')
hold on,
%line([input input],ylim,'color','k','linewidth',2)
if exist('antimodes','var')
gridlines(s_list(antimodes),1,col(2,:),'-')
end
xlabel('intensity')
title('Gradient')
axis 'tight'
axis 'square'
set(gca,'ytick',[])
%-----------------------------------------------------------------