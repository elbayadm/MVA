close all;
clearvars;
sz = [25 50 150 300 500];
rng(0,'twister')
mu1=1; sigma1=1;
mu2=4; sigma2=1;

y1 = normrnd(mu1,sigma1,[500 ,1]);
y2 = {};
for s = sz
    y2{end+1} = normrnd(mu2,sigma2,[s ,1]);
end

figure,
[f,x]= ksdensity(y1);
plot(x,f)
hold on,
[f,x]=ksdensity(y2{1});
plot(x,f)
hold on,
[f,x]=ksdensity(y2{2});
plot(x,f)
hold on,
[f,x]=ksdensity(y2{3});
plot(x,f)
line([2.977 2.977],ylim(),'color','k','linestyle','--')