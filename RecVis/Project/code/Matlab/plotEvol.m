
%% Scores evolution
f0 = figure();
col = [0.93, 0.39, 0.03;0.01, 0.01, 0.84;0.78, 0.01, 0.18;0.04, 0.49, 0.07;0.58, 0.09, 0.62;0.99, 0.84, 0.02;0.19, 0.61, 0.97;0.3, 0.81, 0.53;0.54, 0.18, 0.06];
handles = [];
h1 = area([1 6], [50 50], 'LineStyle',':','LineWidth',1.0, 'FaceColor', [0.87 0.87 0.87],'EdgeColor',[0.5 0.5 0.5]);

hold on
for i=1:9
    h = plot(scores*100, '-o', 'lineWidth',2.0, 'Color', col(i,:));
    handles = [handles h];
    hold on;
end
handles = [handles h1];
f0.Position = [1,1, 827,228];
f0.PaperOrientation = 'landscape';

grid off
box on
legend(handles,{'Im 1','Im 2','Im 3','Im 4','Im 5','Im 6','Im 7','Im 8','Im 9', 'Class 0'},'Location','westoutside','FontSize',12.5);
xlabel('Bootstrap iteration');
ylabel('100xp(label=1)');
set(gca,'FontSize', 12.0);


%% Render images with labels

supergrid = zeros(96*3,96*3,3);
for i = 1:9
   name = sprintf('%d.jpg',i);
   im = imread(name);
   
   x = mod(i-1,3);
   y = idivide(int32(i-1),int32(3));
   supergrid(1+96*x:96*(x+1),1+96*y:96*(y+1),:) = im;
end
f1 = figure();
imshow(supergrid/255);
hold on

for i=1:9
    x = mod(i-1,3);
    y = idivide(int32(i-1),int32(3));
    lab = sprintf('Im %d',i);
    rectangle('Position',[double(96*(x+1) - 28),double(96*(y+1) - 18),25,15],'FaceColor',[0 0 0])
    text(double(96*(x+1) - 25),double(96*(y+1) - 10),lab,'Color',[1 1 1], 'FontSize',11, 'FontWeight','bold')
    hold on;
end

f1.Position = [1 1 500 500];

%% Plot AUC, Acc, Avg Prec. 
close all;
f = figure();

plot(1:7,AUCs(2:8),'-o','Color','b','lineWidth',1.5, 'MarkerFaceColor','b');
hold on;
%plot([1 7], [prec(end) , prec(end)],'--','Color','k','lineWidth',1.5);
hold on;
plot([1 7], [AUCs(end) , AUCs(end)],':','Color','k','lineWidth',1.0);
hold on;
grid on;
box on;
set(gca,'FontSize', 13.0);
xlabel('Bootstrap iteration', 'FontSize',15.0);
ylabel('AUC', 'FontSize',15.0);
legend({'Bootstrap','PerfTr3'},'Location', 'southeast');
f.Position = [1, 1, 1163, 227];
f.PaperOrientation = 'landscape';
f.PaperUnits = 'centimeters';
f.PaperSize = [60 30];
