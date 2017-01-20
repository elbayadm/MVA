%% Settings
clear, close all, clc
warning 'off'
addpath(genpath('Matlab/'))
addpath(genpath('logs/'))
addpath(genpath('features/'))
addpath(genpath('Confusion/'))
addpath('data/morpho_QQ/*')

%% 
close all; clc
%-- Test
test=importdata('test_os.txt');
test_names=test.textdata;
test_labels=test.data(:,1);
test_scores=test.data(:,2);

%-- Full train
full_train=importdata('full_train.txt');
full_train_scores=full_train.data;
full_train_names=full_train.textdata;

%-- Hand labelled training subset
train=importdata('train_os.txt');
train_labels=train.data(:,1);
train_scores=train.data(:,2);

%% Threshold the train set scores and evaluate confusion matrix within the subset

th1=20;
th2=20;
new_labels=-1*ones(length(train_scores),1);
new_labels(train_scores>th2)=1;
new_labels(train_scores<th1)=0;

Q=zeros(2);
Q(1,1)=sum((new_labels==train_labels).*(train_labels==0))/sum((train_labels==0).*(new_labels~=-1));
Q(1,2)=1-Q(1,1);
Q(2,2)=sum((new_labels==train_labels).*(train_labels==1))/sum((train_labels==1).*(new_labels~=-1));
Q(2,1)=1-Q(2,2);
Q
%% Create the new training set:

new_labels=-1*ones(length(full_train_scores),1);
new_labels(full_train_scores>th2)=1;
new_labels(full_train_scores<th1)=0;
% 
file_ = fopen(sprintf('data/morpho_QQ/labels_%d_%d.txt',round(th1),round(th2)),'wt');
fprintf(file_,'%d\n',balanced_list);
fclose(file_);
disp('Done')

% file_ = fopen(sprintf('data/morpho_QQ/images_%d.txt',th),'wt');
% for row = 1:length(new_labels)
%     fprintf(file_,'%s %d\n',subset_names{row},new_labels(row));
% end
% fclose(file_);


%% Balanced set
th1=40;
th2=40;
l1=importdata('labels_40_40.txt');
I0=find(l1==0);
I1=find(l1==1);
sz=length(I1); % To keep
Idrop=I0(randperm(length(I0),length(I0)-sz));
l2=l1;
l2(Idrop)=-1;
file_ = fopen(sprintf('data/morpho_QQ/fifty_labels_%d_%d.txt',round(th1),round(th2)),'wt');
fprintf(file_,'%d\n',l2);
fclose(file_);
disp('Done')

%%
l=importdata('labels_10_30.txt');
list=importdata('balanced_labels_10_30.txt');
sum(l==1)
sum(list==1)
sum(l==0)
sum(list==0)
length(list)
length(l)