%% Settings
clearvars; close all; clc;
warning 'off'
addpath(genpath('Matlab/'))
addpath(genpath('logs/'))
addpath(genpath('features/'))
addpath(genpath('data'))
addpath(genpath('~/Github/toolbox'))
%%
rand('seed',178);
col = [0.93, 0.39, 0.03;0.01, 0.01, 0.84;0.78, 0.01, 0.18;...
    0.04, 0.49, 0.07;0.58, 0.09, 0.62;0.99, 0.84, 0.02;...
    0.19, 0.61, 0.97;0.3, 0.81, 0.53;0.54, 0.18, 0.06]
col=[col;lines(5)];
%% TR1
Tr1 = importdata('Tr1.mat');
s0 = importdata('Tr1_scores.mat');
%%
fd = 'bs3';
p1 = importdata([fd '/sub_strap1_10000.txt']);
p1 = 100*p1(:,2);
s1 = (p1+s0)/2; %2-mean & all-in

p2 = importdata([fd '/sub_strap2_10000.txt']);
p2 = 100*p2(:,2);
%s2 = (p2+s1)/2; %2-mean
s2 = (p2+s1+s0)/3; %all-in

p3 = importdata([fd '/sub_strap3_10000.txt']);
p3 = 100*p3(:,2);
%s3 = (p3+s2)/2; %2-mean
s3 = (p3+s2+s1+s0)/4; % all-in
p4 = importdata([fd '/sub_strap4_10000.txt']);
p4 = 100*p4(:,2);
%s4 = (p4+s3)/2; %2-mean
s4 = (p4+s3+s2+s1+s0)/5;

p5 = importdata([fd '/sub_strap5_10000.txt']);
p5 = 100*p5(:,2);
%s5 = (p5+s4)/2; %2-mean
s5 = (p5+s4+s3+s2+s1+s0)/6; %all-in
p6 = importdata([fd '/sub_strap6_10000.txt']);
p6 = 100*p6(:,2);

%%
%pick = randi([1 4152],[12 1]);
%pick = [2301  3124  4045 456 2313 1402]; %pos 3727 4025 2122 2944 362 46 2028
%pick = [ 802  1297 3464 2391 2687 2676];  %neg 391 2769 1339
pick = [4025 2122 2944 362 46 2028 2301]; % pos bis
scores = [s0(pick) s1(pick) s2(pick) s3(pick) s4(pick) s5(pick)];
proba = [p1(pick) p2(pick) p3(pick) p4(pick) p5(pick) p6(pick)];

figure(1); clf;
set(1,'units','normalized','position',[.1 .1 .5 .25])
set(1,'PaperType','A4','PaperPositionMode','auto') ;
hold on,
h1 = area([0 0.5], [30 30], 'LineStyle',':','LineWidth',1.0, 'FaceColor', [0.87 0.87 0.87],'EdgeColor',[0.5 0.5 0.5]);
h2 = area([0.5 1.5], [40 40], 'LineStyle',':','LineWidth',1.0, 'FaceColor', [0.87 0.87 0.87],'EdgeColor',[0.5 0.5 0.5]);
h3 = area([1.5 5], [50 50], 'LineStyle',':','LineWidth',1.0, 'FaceColor', [0.87 0.87 0.87],'EdgeColor',[0.5 0.5 0.5]);
handles = [];
for i=1:length(pick)
    handles(end+1)= plot(0:5,scores(i,:),'-o','color',col(i,:),'linewidth',1.5);
end
handles(end+1)=h1;
labs = cellstr(num2str((1:length(pick))', 'Im%d'));
labs{end+1}='class 0';
legend(handles,labs,'location','westoutside','FontSize',12.5)
ylim([0 100])
grid off
box on
xlabel('Bootstrap 1 iteration');
ylabel('Score');
set(gca, 'Layer', 'top')

figure(2);clf
set(2,'units','normalized','position',[.1 .1 .213 .23])
for i=1:length(pick)
    vl_tightsubplot(length(pick),i,'box','inner') ;
    imagesc(squeeze(Tr1(pick(i),:,:,:))) ;
    text(65,70,sprintf('Im %d',i),...
       'color','w',...
       'background','k',...
       'verticalalignment','top', ...
       'fontsize', 10) ;
   set(gca,'xtick',[],'ytick',[]) ; axis image ;
end

figure(3); clf;
set(3,'units','normalized','position',[.1 .1 .5 .25])
set(3,'PaperType','A4','PaperPositionMode','auto') ;
hold on,
h1 = area([1 6], [50 50], 'LineStyle',':','LineWidth',1.0, 'FaceColor', [0.87 0.87 0.87],'EdgeColor',[0.5 0.5 0.5]);
handles = [];
for i=1:length(pick)
    handles(end+1)=plot(1:6,proba(i,:),'-o','color',col(i,:),'linewidth',1.5);
end
handles(end+1)= h1;
labs = cellstr(num2str((1:length(pick))', 'Im%d'));
labs{end+1}='class 0';
legend(handles,labs,'location','westoutside','FontSize',12.5)
ylim([0 100])
grid off
box on
xlabel('Bootstrap 1 iteration');
ylabel('100xp(label=1)');
set(gca, 'Layer', 'top')
%print '-f1' 'figures/rapport/tocrop/bs3_scores_pos' '-dpdf'
%print '-f2' 'figures/rapport/tocrop/bs2_images_pos' '-dpdf'
%print '-f3' 'figures/rapport/tocrop/bs3_probas_pos' '-dpdf'