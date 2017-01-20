%% Settings
clear, close all, clc
warning 'off'
addpath(genpath('Matlab/'))
addpath(genpath('logs/'))
addpath(genpath('features/'))
addpath('data/morpho_QQ/original')
addpath('data/morpho_QQ/cleanMerge')
addpath('data/morpho_QQ/clean')
%% Files:
% % Old scores:
% full_train=importdata('full_train.txt');
% full_train_scores=full_train.data;
% full_train_names=full_train.textdata;
% lfw=importdata('out_glasses_lfw.txt');
% lfw_names=lfw.textdata;
% lfw_labels=lfw.data;
% clear lfw
%Train subset:
sub=importdata('test_os.txt');
sub_names=sub.textdata;
sub_labels=sub.data(:,1);
sub_scores=sub.data(:,2);
clear sub

file_ = fopen('data/morpho_QQ/clean/test_scores.txt','wt');
fprintf(file_,'%d\n',floor(sub_scores));
fclose(file_);
disp('Done')

%% balance the sub then merge:

I=find(sub_labels==1);
J=find(sub_labels==0);
% balance:
p=4;
J=J(1:(4*length(I)));
ind=sort([I;J]);
sub_names=sub_names(ind);
sub_labels=sub_labels(ind);
%%
names=[lfw_names;sub_names];
labels=[lfw_labels;sub_labels];
% shuffle:
perm=randperm(length(names),length(names));
names=names(perm);
labels=labels(perm);
%
%% Write sub-images
file_ = fopen('data/morpho_QQ/cleanMerge/clean1.txt','wt');
for i=1:(length(labels))
    fprintf(file_,'%s ',names{i});
    fprintf(file_,'%d\n',labels(i));
end
fclose(file_);
disp('Done')
%% Decrease size:
list=importdata('clean1.txt');
p=.9;
labels=list.data;
for i=2:10
    new_labels=labels;
    for j=1:length(labels)
        if rand(1)>p %ignore it
            new_labels(j)=-1;
        end
    end
    fprintf('Lmdb %d : pos(%d) neg(%d) size(%d)\n',i,sum(new_labels==1),sum(new_labels==0),sum(new_labels~=-1));
    file_ = fopen(sprintf('data/morpho_QQ/cleanMerge/clean%d.txt',i),'wt');
    fprintf(file_,'%d\n',new_labels);
    fclose(file_);
    labels=new_labels;
    p=p*.9;
end
%% from 7 to 10 convert to imageset
images=importdata('clean1.txt');
images=images.textdata;
for i=7:10
list=importdata(sprintf('clean%d.txt',i));
I=find(list==1);
J=find(list==0);
subi=images(sort([I;J]));
subl=list(sort([I;J]));
file_ = fopen(sprintf('data/morpho_QQ/cleanMerge/im_clean%d.txt',i),'wt');
for j=1:length(subl)   
fprintf(file_,'%s %d\n',subi{j},subl(j));
end
fclose(file_);
end
%% Results:
rand('seed',178);
col=[
    
    50 155 247;%bleu ciel
    237 99 7;%Dark orange
    2 2 214;%dark blue
    145 145 145;%Gris
    199 2 45; %brickred
    75 247 45;%jade
    227 82 200;%Dusty pink
    9 125 18;%flash green
    147 22 158; %Violet
    252 215 5%Gold
    
    ]/255;
%col=[col;.8*hsv(12);summer(12)];
%col=col(randperm(size(col,1),size(col,1)),:);
labels=importdata('test_labels.txt');
labels=2*labels.data-1;
%% logs & features
f={};
sz=[];
pos=[];
acc=[];
% 20%
for i=1:10
    disp(i)
    l=importdata(sprintf('morpho/clean_v2/clean%d.log.test',i));
    acc(end+1)=l.data(end,4);
    f{end+1}=importdata(sprintf('morpho/clean_v2/clean%d.txt',i));
    if i==1
        list=importdata(sprintf('cleanMerge/clean%d.txt',i));
        list=list.data;
        pos(end+1)=sum(list==1);
        sz(end+1)=sum(list~=-1);
    elseif i<8
        list=importdata(sprintf('cleanMerge/clean%d.txt',i));
        pos(end+1)=sum(list==1);
        sz(end+1)=sum(list~=-1);
    else
        list=importdata(sprintf('im_clean%d.txt',i));
        list=list.data;
        pos(end+1)=sum(list==1);
        sz(end+1)=length(list);
    end
end
% 5%
for i=1:19
    disp(i)
    l=importdata(sprintf('morpho/clean_v1/clean%d.log.test',i));
    acc(end+1)=l.data(end,4);
    f{end+1}=importdata(sprintf('morpho/clean_v1/clean%d.txt',i));
    if i==1
        list=importdata('subtrain_images.txt');
        list=list.data;
        pos(end+1)=sum(list==1);
        sz(end+1)=sum(list~=-1);   
    else
        list=importdata(sprintf('clean/labels_clean%d.txt',i));
        pos(end+1)=sum(list==1);
        sz(end+1)=sum(list~=-1);
    end
end
f{14}=f{14}(1:4661,:);
LS=[2 5 10 20 50 100 150 200 250 300 350]*1000;
LS=fliplr(LS);
for i=1:length(LS)
    disp(i)
    f{end+1}=importdata(sprintf('morpho/noisy/%d.txt',LS(i)));
    sz(end+1)=LS(i);
end
clear list l
%% Accuracy
dim=10;
figure(1), clf
set(1,'units','normalized','position',[.1 .1 .6 .3])
set(1,'PaperType','A4','PaperPositionMode','auto','PaperOrientation','landscape')
semilogx(sz(1:dim),acc(1:dim),'-xb');
title('Accuracy: #samples')
xlabel('Training set size')
ylabel('Accuracy')
xlim([min(sz(1:dim)),max(sz(1:dim))])
grid on
grid minor
%print 'figures/clean2/acc_size_log' '-dpdf'
%% Roc crves:
% figure(2), clf;
% set(2,'units','normalized','position',[.1 .1 .7 .7])
% set(2,'PaperType','A4','PaperPositionMode','auto')
close all; clc;
dim=length(sz);
legends={'random'};
auc=zeros(1,dim);
ap=zeros(1,dim);
ar=zeros(1,dim);
pick=[5 7 11 17 19 35 40];
c=1;
for i=1:dim
    flag=0;
    fprintf('%d..',i);
    [scores,pred]=max(f{i},[],2);
    pred=2*pred-3;
    scores=scores.*pred;
   % [tpr,tnr,info]=vl_pr(labels,scores);
    if i==2
        prec_rec(scores,labels,'numThresh',8000,'plotROC',1,'plotPR',0,'col',col(c,:));
        c=c+1;
        flag=1;
    end
    if ismember(i,pick)
        prec_rec(scores,labels,'numThresh',8000,'plotROC',1,'plotPR',0,'col',col(c,:),'holdFigure',1);
        flag=1;
        c=c+1;
    end
    [PREC, TPR, FPR, THRESH] = prec_rec(scores,labels,'numThresh',8000,'plotROC',0,'plotPR',0);
    [~,~,info]=vl_pr(labels,scores);
    ap(i)=info.ap;
    ar(i)=mean(TPR);
    [tpr,tnr,info]=vl_roc(labels,scores);
    auc(i)=info.auc;
    if flag
        fprintf('\n');
        if i<30
            legends{end+1}=sprintf('clean size %d - %.0f%%',sz(i),pos(i)/sz(i)*100);
        else
            legends{end+1}=sprintf('noisy size %d - 7.5%%',sz(i));
        end
    end
end
axis square
grid on,
% ylim([.5 1])
% xlim([0 .5])
% xlabel('false positives rate')
% ylabel('true positives rate')
legend(legends,'fontsize',12,'location','eastoutside');
print 'figures/rapport/tocrop/roc_size' '-dpdf'
%%
figure(3), clf
set(3,'units','normalized','position',[.1 .1 .6 .5])
set(3,'PaperType','A4','PaperPositionMode','auto','PaperOrientation','landscape')
subplot(2,1,1)
l1=semilogx([sz(1:10) sz(11:21) sz(30:end)],[ap(1:10) ap(11:21) ap(30:end)],'ob','MarkerSize',5,'MarkerFaceColor','b');
hold on,
l2=semilogx(sz(1:10),ap(1:10),'-ob','MarkerSize',5,'MarkerFaceColor','b');
hold on,
l3=semilogx(sz(11:21), ap(11:21),'-or','MarkerSize',5,'MarkerFaceColor','r');
hold on,
l4=semilogx(sz(30:end), ap(30:end),'-o','color',col(8,:),'MarkerSize',5,'MarkerFaceColor',col(8,:));
line([1 1e6],[0.9537 0.9537],'linestyle','--','color','k','linewidth',2);
line([1 1e6],[0.6842 0.6842],'linestyle',':','color','k','linewidth',2);
legend([l2,l3,l4],{'clean 20%','clean 5%','noisy'},'location','southwest')
title('Average precision')
ylabel('Average precision')
grid on,
grid minor,
%
subplot(2,1,2)
ex=importdata('subnoisy_20pc.txt');
ex=ex.data;
l1=semilogx([sz(1:10) sz(11:21) sz(30:end)],[auc(1:10) auc(11:21) auc(30:end)],'ob','MarkerSize',5,'MarkerFaceColor','b');
hold on,
l2=semilogx(sz(1:10),auc(1:10),'-ob','MarkerSize',5,'MarkerFaceColor','b');
hold on,
l3=semilogx(sz(11:21), auc(11:21),'-or','MarkerSize',5,'MarkerFaceColor','r');
hold on,
l4=semilogx(sz(30:end), auc(30:end),'-o','color',col(8,:),'MarkerSize',5,'MarkerFaceColor',col(8,:));
hold on,
l5=semilogx(ex(:,1), ex(:,2),'-o','color',col(7,:),'MarkerSize',5,'MarkerFaceColor',col(7,:));
l6=line([1 1e6],[.9878 0.9878],'linestyle','--','color','k','linewidth',2);
l7=line([1 1e6],[.9196 0.9196],'linestyle',':','color','k','linewidth',2);
title('AUC')
xlabel('Training set size')
ylabel('AUC')
legend([l2,l3,l4,l5],{'clean 20%','clean 5%','noisy 5%','noisy 20%'},'location','southwest')
grid on,
grid minor,
ah=axes('position',get(gca,'position'),'visible','off');
legend(ah,[l6,l7],{'perfTr2','perfOld'},'location','southeast')
print 'figures/rapport/tocrop/ap_auc' '-dpdf'
%%
sz=[];
p=[];
for i=1:10
    list=importdata(sprintf('clean%d.txt',i));
    if i==1
        list=list.data;
    end
    sz(end+1)=sum(list~=-1);
    p(end+1)=sum(list==1)/sum(list~=-1);
end
fprintf('%d &',sz);
disp('\\')
fprintf('%.2f\\%% &',p*100);