%% Section 2.1: subgradient method:
rng(0,'twister');
colors=jet(8);
doprint=true;

lambda=10;
eps=1e-5;
n=100;
d=10;

X=rand(n,d);
y=rand(n,1);
%% Plot the results for different strategies:
%% constant step size
h=[10 5 1 .5 .1]*1e-4;
figure('units','normalized','position',[0 0 .7 .3],'PaperPositionMode','auto','PaperOrientation','landscape'),
set(0,'DefaultAxesFontSize',14)
for i=1:length(h)
fprintf('h=%.0e\n',h(i));
[w_seq , loss_seq, loss_optim]=subgrad(X,y,lambda,eps,'constant',h(i));
subplot(1,2,1)
plot(loss_seq,'color',colors(i,:))
hold on,
subplot(1,2,2)
semilogx(loss_optim-min(loss_optim),'linewidth',2,'color',colors(i,:))
hold on,
end
subplot(1,2,1)
title('constant size');
ylabel('f(w^{(k)})')
xlabel('iteration')
xlim([1 size(w_seq,2)])

subplot(1,2,2)
title('constant size');
legend(cellstr(num2str(h', 'h=%.0e')))
ylabel('f_{best}^{(k)}-p^*')
xlabel('iteration')
xlim([1 size(w_seq,2)])

if doprint
print('-dpdf','images/sub_cst.pdf', '-opengl')
end

%% constant step length
h=[10 5 1 .5 .1]*1e-2;
figure('units','normalized','position',[0 0 .7 .3],'PaperPositionMode','auto','PaperOrientation','landscape'),
for i=1:length(h)
fprintf('h=%.0e\n',h(i));
[w_seq , loss_seq, loss_optim]=subgrad(X,y,lambda,eps,'length',h(i));
subplot(1,2,1)
plot(loss_seq,'--','linewidth',.6,'color',colors(i,:))
hold on,
subplot(1,2,2)
semilogx(loss_optim-min(loss_optim),'linewidth',2,'color',colors(i,:))
hold on,
end
subplot(1,2,1)
title('constant length');
ylabel('f(w^{(k)})')
xlabel('iteration')
xlim([1 size(w_seq,2)])

subplot(1,2,2)
title('constant length');
legend(cellstr(num2str(h', 'h=%.0e')))
ylabel('f_{best}^{(k)}-p^*')
xlabel('iteration')
xlim([1 size(w_seq,2)])

if doprint
print('-dpdf','images/sub_length.pdf', '-opengl')
end
%% Square summable but not summable
h=[10 5 1 .5 .1]*1e-3;
figure('units','normalized','position',[0 0 .7 .3],'PaperPositionMode','auto','PaperOrientation','landscape'),
for i=1:length(h)
fprintf('h=%.0e\n',h(i));
[w_seq , loss_seq, loss_optim]=subgrad(X,y,lambda,eps,'sqs',h(i));
subplot(1,2,1)
plot(loss_seq,'--','linewidth',2,'color',colors(i,:))
hold on,
subplot(1,2,2)
semilogx(loss_optim-min(loss_optim),'linewidth',2,'color',colors(i,:))
hold on,
end
subplot(1,2,1)
title('L2\L1');
ylabel('f(w^{(k)})')
xlabel('iteration')
xlim([1 size(w_seq,2)])

subplot(1,2,2)
title('L2\L1');
legend(cellstr(num2str(h', 'h=%.0e')))
ylabel('f_{best}^{(k)}-p^*')
xlabel('iteration')
xlim([1 size(w_seq,2)])

if doprint
print('-dpdf','images/sub_sqs.pdf', '-opengl')
end
%% dimnishing nonsummable 
h=[10 5 1 .5 .1]*1e-3;
figure('units','normalized','position',[0 0 .7 .3],'PaperPositionMode','auto','PaperOrientation','landscape'),
for i=1:length(h)
fprintf('h=%.0e\n',h(i));
[w_seq , loss_seq, loss_optim]=subgrad(X,y,lambda,eps,'diminish',h(i));
subplot(1,2,1)
plot(loss_seq,'--','linewidth',2,'color',colors(i,:))
hold on,
subplot(1,2,2)
semilogx(loss_optim-min(loss_optim),'linewidth',2,'color',colors(i,:))
hold on,
end
subplot(1,2,1)
title('Diminishing');
ylabel('f(w^{(k)})')
xlabel('iteration')
xlim([1 size(w_seq,2)])

subplot(1,2,2)
title('Diminishing');
legend(cellstr(num2str(h', 'h=%.0e')))
ylabel('f_{best}^{(k)}-p^*')
xlabel('iteration')
xlim([1 size(w_seq,2)])

if doprint
print('-dpdf','images/sub_diminish.pdf', '-opengl')
end
%% Section 2.2: Coordinate descent method:
[mu_seq, w_seq, loss_seq, dGap]= coord_descent(X,y,lambda,eps);
figure('units','normalized','position',[0 0 .7 .3],'PaperPositionMode','auto','PaperOrientation','landscape'),
subplot(1,2,1)
semilogx(loss_seq-min(loss_seq),'-*r')
xlim([1 length(loss_seq)]);
xlabel('iteration')
ylabel('f(w^{(k)})-d^*')
title('coordinate descent - convergence')
subplot(1,2,2)
semilogx(dGap,'-*b')
xlim([1 length(loss_seq)]);
xlabel('iteration')
ylabel('gap')
title('coordinate descent - duality gap')

if doprint
print('-dpdf','images/coord.pdf', '-opengl')
end

%% CPU Timing
coord_time=[];
sub_cst_time=[];
sub_length_time=[];
sub_sqs_time=[];
sub_diminish_time=[];
proximal_time=[];
proximal_acc_time=[];

coord_iter=[];
sub_cst_iter=[];
sub_length_iter=[];
sub_sqs_iter=[];
sub_diminish_iter=[];
proximal_iter=[];
proximal_acc_iter=[];

for eps=[1e-3 1e-6 1e-10]
    fprintf('Coordinate descent\n');
    tic,
    [~ , ~ , ~, ~, iter]= coord_descent(X,y,lambda,eps);
    coord_time=[coord_time toc];
    coord_iter=[coord_iter iter];
    
    fprintf('Subgradient- constant\n');
    tic,
    [~ , ~, ~,iter]=subgrad(X,y,lambda,eps,'constant',1e-3);
    sub_cst_time=[sub_cst_time toc];
    sub_cst_iter=[sub_cst_iter iter];
    
    fprintf('Subgradient- length\n');
    tic,
    [~ , ~, ~,iter]=subgrad(X,y,lambda,eps,'length',1e-2);
    sub_length_time=[sub_length_time toc];
    sub_length_iter=[sub_length_iter iter];
    
    fprintf('Subgradient- sqs\n');
    tic,
    [~ , ~, ~,iter]=subgrad(X,y,lambda,eps,'sqs',1e-2);
    sub_sqs_time=[sub_sqs_time toc];
    sub_sqs_iter=[sub_sqs_iter iter];
    
    fprintf('Subgradient- diminish\n');
    tic,
    [~ , ~, ~,iter]=subgrad(X,y,lambda,eps,'diminish',5e-3);
    sub_diminish_time=[sub_diminish_time toc];
    sub_diminish_iter=[sub_diminish_iter iter];
    
    %Section 3:
    fprintf('Proximal\n');
    tic,
    [~,~, iter]=prox_method(X,y,lambda,eps);  
    proximal_time=[proximal_time toc];
    proximal_iter=[proximal_iter iter];
    
    fprintf('accelerated proximal\n');
    tic,
    [~,~, iter]=prox_acc_method(X,y,lambda,eps);  
    proximal_acc_time=[proximal_acc_time toc];
    proximal_acc_iter=[proximal_acc_iter iter];
    
end
coord_time
sub_cst_time
sub_length_time
sub_sqs_time
sub_diminish_time
proximal_time
coord_iter
sub_cst_iter
sub_length_iter
sub_sqs_iter
proximal_iter
sub_diminish_iter