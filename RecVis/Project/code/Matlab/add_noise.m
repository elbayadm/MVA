%% Settings
clear, close all, clc
warning 'off'
addpath(genpath('Matlab/'))
addpath(genpath('logs/'))
addpath(genpath('features/'))
addpath(genpath('Confusion/'))
addpath('data/morpho_QQ')

%% shuffle 'once'
%%--
% full_train=importdata('scores.txt');
% train_names=full_train.textdata;
% I=ismember(train_names,test_names);
% full_train_names=train_names(~I);
% full_train_scores=full_train.data(~I,:);
% shuffle = randperm(length(full_train_scores));
% full_train_names=full_train_names(shuffle);
% full_train_scores=full_train_scores(shuffle);
% 
% file_ = fopen(sprintf('data/morpho_QQ/full_train.txt',th),'wt');
% for row = 1:length(full_train_names)
%     fprintf(file_,'%s %.2f\n',full_train_names{row},full_train_scores(row));
% end
% fclose(file_);
% disp('Done')
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

th1=30;
th2=30;
new_labels=-1*ones(length(train_labels),1);
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

file_ = fopen(sprintf('data/morpho_QQ/labels_%d_%d.txt',round(th1),round(th2)),'wt');
fprintf(file_,'%d\n',new_labels);
fclose(file_);
disp('Done')

% file_ = fopen(sprintf('data/morpho_QQ/images_%d.txt',th),'wt');
% for row = 1:length(new_labels)
%     fprintf(file_,'%s %d\n',subset_names{row},new_labels(row));
% end
% fclose(file_);


%% Stats on file:
clc
list=importdata('labels_40_40.txt');
length(list)
sum(list~=-1)
sum(list==1)
sum(list==0)
