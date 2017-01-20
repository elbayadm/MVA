[Dtrain,Dtest]  = load_digit7;
C = lines(10);
%%
figure(1)
%The first k-means
rng(1,'twister');
km=2;
[centers,distortion,affect]=kmeans(Dtrain,km);
plot(log(distortion),'color',C(1,:),'marker','x','linewidth',2)
xlim([1 length(distortion)])
ylim([9.78 9.8])
xlabel('k-means iteration','fontsize',20);
ylabel('log(Distortion)','fontsize',20);
%hold on,

%Keeping the best:
% for k=2:10
%     [new_centers,new_distortion,new_affect]=kmeans(Dtrain,km);
%     if distortion(end)>new_distortion(end)
%         centers=new_centers;
%         distortion=new_distortion;
%         affect=new_affect;
%     end
%     plot(new_distortion,'color',C(k,:),'marker','x','linewidth',2)
%     hold on,
% end
print(1,'-dpdf','images/tocrop/km_dist.pdf', '-opengl');


%% Showing the centroids

figure(2)
nr=1;
nc=floor(km/nr)+1*(mod(km,nr)~=0);
for k=1:km
    subplot(nr,nc,k)
    c_image=reshape(centers(k,:),[28,28]);
    imshow(c_image);
    t=title(sprintf('Centroid %d',k));
    set(t,'FontSize',14, 'FontWeigh','normal');
end
print(2,'-dpdf','images/tocrop/km_centroids.pdf', '-opengl');

C1=datasample(Dtrain(affect==1,:),15,'replace',false);
figure(3)
for k=1:15
    vl_tightsubplot(3,5,k,'box','inner') ;
    c_image=reshape(C1(k,:),[28,28]);
    imshow(c_image);
end
print(3,'-dpdf','images/tocrop/km_c1.pdf', '-opengl');

C2=datasample(Dtrain(affect==2,:),15,'replace',false);
figure(4)
for k=1:15
    vl_tightsubplot(3,5,k,'box','inner') ;
    c_image=reshape(C2(k,:),[28,28]);
    imshow(c_image);
end
print(4,'-dpdf','images/tocrop/km_c2.pdf', '-opengl');

%% Varying the number of clusters k:
Ds_train=zeros(1,6);
Ds_test=zeros(1,6);
km=[3, 4, 5, 10, 50, 100];
for i=1:length(km)
   [centers,distortion,affect]=kmeans(Dtrain,km(i));
   fprintf('@k=%d\n',km(i));
%Keeping the best:
%   for k=2:10
%     [new_centers,new_distortion,new_affect]=kmeans(Dtrain,km(i));
%     if distortion(end)>new_distortion(end)
%         centers=new_centers;
%         distortion=new_distortion;
%         affect=new_affect;
%     end
%   end
    Ds_train(i)=distortion(end);
    [Ds_test(i),~]=km_distortion(Dtest,centers);
end

%%
figure(5)
plot(km,log(Ds_train),'color',C(1,:),'marker','x','linewidth',2)
hold on,
plot(km,log(Ds_test),'color',C(2,:),'marker','x','linewidth',2)
xlabel('k');
ylabel('log(Distortion)','fontsize',20);
legend('Train','Test');
print(5,'-dpdf','images/tocrop/km_train_test.pdf', '-opengl');