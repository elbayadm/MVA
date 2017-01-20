%% Settings:
clearvars;close all;clc;
set(0,'DefaultAxesFontSize',14)
set(0,'DefaultTextFontSize',14)

% Loading toolboxes:
addpath(genpath('matlab-toolboxes/toolbox_signal'))
addpath(genpath('matlab-toolboxes/toolbox_general'))

%% -------------------------
%% Sliced Optimal Transport: 
%% -------------------------
%% Wasserstein Distance
%% 1-D Optimal Transport
%% Wasserstein Projection
options.rows=1;
P = @(f,g)perform_hist_eq(f,g,options);
%% Sliced Wasserstein Distanc
%% Sliced Wasserstein Projection
%% Matching of 2-D Distributions
N = 300;
d = 2;
% square
f = rand(2,N)-.5;
% anulus
theta = 2*pi*rand(1,N);
r = .8 + .2*rand(1,N);
g = [cos(theta).*r; sin(theta).*r];

% display shortcut
plotp = @(x,col)plot(x(1,:)', x(2,:)', 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', col, 'LineWidth', 2);
figure(1);clf;
plotp(f, 'b')
hold on
plotp(g, 'r')
axis('equal')
axis('off')

%% Gradient descent
% stepsize 
tau = .2;
% Initialization
f1 = f;
% Orthogonal coordinate system
[Theta,~] = qr(randn(d));
% update:
f1 = (1-tau)*f1 + tau * Theta * P(Theta'*f1, Theta'*g);
[Theta,~] = qr(randn(d));

figure(1);clf;
plotp(f, 'b')
hold on
plotp(g, 'r')
hold on
plotp(f1, 'y')
axis('off'); axis('equal');

%% Exo 1
niter = 1000;
figure(1); clf;
f1 = f;
disp_list = [1 3 10 100]; q = 1;
for i=1:niter    
    [Theta,~] = qr(randn(d));    
    f1 = (1-tau)*f1 + tau * Theta * P(Theta'*f1, Theta'*g);
    if q<=4 && i==disp_list(q)
        t = (q-1)/3;
        subplot(2,2,q);
        hold on;
        plotp(f1, [t 0 1-t]);
        axis('off'); axis('equal');
        q = q+1;
    end
end
%% Final configuration
figure(1); clf;
hold on;
h = plot([f(1,:);f1(1,:)], [f(2,:);f1(2,:)], 'k');
set(h, 'LineWidth', 2);
plotp(f, 'b');
plotp(g, 'r');
axis('off'); axis('equal');

%% Barycentres
t = .5;
ft = (1-t)*f + t*f1;
figure(1);clf;
hold on;
plotp(f, 'b');
plotp(g, 'r');
plotp(ft, [t 0 1-t]);
axis('off'); axis('equal');

%% Progressive interpolation
figure(1);clf;
ts= linspace(0 , 1, 6);
for i = 1:length(ts)
    t = ts(i);
    ft = (1-t)*f + t*f1;
    subplot(2,3,i)
        plotp(f, 'b');
        plotp(g, 'r');
        plotp(ft, [t 0 1-t]);
        axis('off'); axis('equal');
        title(sprintf('t = %.2f',t))
end

%% 3D Histogram Matching
n = 128;
N = n*n;
d = 3;
F = rescale( load_image('hibiscus', n) );
G = rescale( load_image('flowers', n) );
figure(1);clf;
imageplot({F G});

%% 2D-histograms
f = reshape(F, [n*n 3])';
g = reshape(G, [n*n 3])';
quantize = @(A,Q)1+round((Q-1)*A);
J = @(I,Q)I(1,:)' + Q*(I(2,:)'-1);
hist2d = @(f,Q)reshape( accumarray(J(quantize(f,Q),Q), ones(1,N), [Q*Q 1], @sum), [Q Q]);
Q = 60;
func = @(a)log(a+3);
figure(1); clf;
imageplot({ func(hist2d(f(1:2,:),Q)), func(hist2d(g(1:2,:),Q)) });

%% Exo 3: Per channel equalization
f1 = P(f,g);
F1 = reshape(f1', [n n 3]);
figure(1); clf;
imageplot(F1);

%% Exo 4: minimizing Sliced Wasserstein
niter = 1000;
figure(1); clf;
f1 = f;
q = 1; disp_list = [3 10 100 niter];
for i=1:niter    
    [Theta,~] = qr(randn(d));    
    f1 = (1-tau)*f1 + tau * Theta * P(Theta'*f1, Theta'*g);
    if q<=4 && i==disp_list(q)
        subplot(2,2,q);
        F1 = reshape(f1', [n n 3]);
        imageplot(F1,sprintf('iter %d',i));
        q = q+1;
    end
end
%
F1 = reshape(f1', [n n 3]);
figure(2); clf;
imageplot(F1,'Final image');

%% Exo 5: geodesic interpolation

tlist = linspace(0,1,6);
figure(1); clf;
for i=1:length(tlist)
    t = tlist(i);
    ft = (1-t)*f + t*f1;
    subplot(2,length(tlist)/2,i);
    imageplot( func(hist2d(ft(1:2,:),Q)) );
    title(['t=' num2str(t,2)]);
end
