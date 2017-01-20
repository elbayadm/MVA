%% Settings
clearvars; close all; clc;
warning 'off'
addpath(genpath('Matlab/'))
addpath(genpath('logs/'))
addpath(genpath('features/'))
addpath('data/original')
addpath(genpath('~/Github/toolbox'))
%%
rand('seed',178);
col=[
    9 125 18;%Green
    97 6 158;%Violet
    199 2 45; %brickred
    2 2 214;%dark blue
    237 99 7;%Dark orange
    145 145 145;%Gris
    127 245 163;%Light green
    252 215 5%Gold
    50 155 247;%bleu ciel
]/255;
%% B37
M1_37 = importdata('roc_cnn_B37_Bootstrap_avg_iter5.txt');
M2_37 = importdata('roc_cnn_B37_Bootstrap_init_hk_iter3.txt');
M3_37 = importdata('roc_cnn_B37_Tr3_iter_10000.txt');
figure(1);clf;
loglog(M1_37(:,1),M1_37(:,2),'color',col(1,:),'linewidth',2)
hold on,
loglog(M2_37(:,1),M2_37(:,2),'color',col(2,:),'linewidth',2)
hold on,
loglog(M3_37(:,1),M3_37(:,2),'color',col(3,:),'linewidth',2)

%% B53
M1_53 = importdata('roc_cnn_B53_Bootstrap_avg_iter5.txt');
M2_53 = importdata('roc_cnn_B53_Bootstrap_init_hk_iter3.txt');
M3_53 = importdata('roc_cnn_B53_Tr3_iter_10000.txt');
legs={};
figure(1);clf;
semilogx(M1_53(:,1),M1_53(:,2),'color',col(1,:),'linewidth',2)
legs{end+1}= sprintf('M1| AUC = %.3f',-trapz(M1_53(:,1),M1_53(:,2)));

hold on,
semilogx(M2_53(:,1),M2_53(:,2),'color',col(2,:),'linewidth',2)
legs{end+1}= sprintf('M2| AUC = %.3f',-trapz(M2_53(:,1),M2_53(:,2)));

hold on,
semilogx(M3_53(:,1),M3_53(:,2),'color',col(3,:),'linewidth',2)
legs{end+1}= sprintf('M3| AUC = %.3f',-trapz(M3_53(:,1),M3_53(:,2)));
legend(legs,'location','southeast')
axis 'square'
grid on
xlabel('log(false positives rate')
xlabel('true positives rate')
title('B53 -  (size = 807 , |positives|=146)')
legend
% xlim(log([0,1]))
% ylim([0,1])

%% B61
M1_61 = importdata('roc_cnn_B61_Bootstrap_avg_iter5.txt');
M2_61 = importdata('roc_cnn_B61_Bootstrap_init_hk_iter3.txt');
M3_61 = importdata('roc_cnn_B61_Tr3_iter_10000.txt');
figure(1);clf;
semilogx(M1_61(:,1),M1_61(:,2),'color',col(1,:),'linewidth',2)
hold on,
semilogx(M2_61(:,1),M2_61(:,2),'color',col(2,:),'linewidth',2)
hold on,
semilogx(M3_61(:,1),M3_61(:,2),'color',col(3,:),'linewidth',2)
axis 'square'
% xlim(log([0,1]))
% ylim([0,1])

%% 
trapz(M1_61(:,1),M1_61(:,2))
