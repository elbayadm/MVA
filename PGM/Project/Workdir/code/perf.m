
close all; 
%------------------------------------
disp('Top N Most confident detections')

correct_baseline = zeros([10 1]);
counts_baseline = zeros([10 1]);
correct_model = zeros([10 1]);
counts_model = zeros([10 1]);
Ncorrect_baseline = zeros(M,1);
Ncorrect_model = zeros(M,1);
Ncorrect_bound = zeros(M,1);
for n = 1:M
    % baseline
    scores_baseline = [DdetectortestUpdate(n).annotation.object.p_w_s];
    type_baseline = ismember({DdetectortestUpdate(n).annotation.object.detection}, {'correct'});
    
    [s,k] = sort(scores_baseline, 'descend');
    Nd = length(k);
    Nd = min(Nd,10);
    correct_baseline(1:Nd) = correct_baseline(1:Nd)+type_baseline(k(1:Nd))';
    counts_baseline(1:Nd) = counts_baseline(1:Nd)+1;
    Ncorrect_baseline(n) = find(type_baseline(k)==0, 1 )-1;    
    
    % model
    scores_model = [DdetectortestUpdate(n).annotation.object.confidence];
    type_model = ismember({DdetectortestUpdate(n).annotation.object.detection}, {'correct'});
    
    [s,k] = sort(scores_model, 'descend');
    Nd = length(k); 
    Nd = min(Nd,10);
    %Nd = min(Nd, sum(type_baseline));
    correct_model(1:Nd) = correct_model(1:Nd)+type_model(k(1:Nd))';
    counts_model(1:Nd) = counts_model(1:Nd)+1;
    Ncorrect_model(n) = find(type_model(k)==0, 1 )-1;
    
    % ground truth
    objects = {test_imdb(n).annotation.object.name};
    [foo,true_obj] = ismember(objects, names); 
    Ncorrect_bound(n) = length(find(true_obj>0));        
end
perf_basline = correct_baseline./counts_baseline;
perf_model = correct_model./counts_model;


figure
subplot(121)
h_t = hist(Ncorrect_bound, 0:5);
ht = cumsum(h_t(6:-1:2));
h_b = hist(Ncorrect_baseline, 0:5);
hb = cumsum(h_b(6:-1:2))./ht;
h_m = hist(Ncorrect_model, 0:5);
hm = cumsum(h_m(6:-1:2))./ht;
bar(([hb(end:-1:1); hm(end:-1:1)])')
title('Localization')
axis('square')
xlabel('N')
ylabel('Percentage of Images')
grid on
legend('Baseline','Context')

% Presence prediction
correct_baseline2 = zeros([10 1]);
counts_baseline2 = zeros([10 1]);
correct_model2 = zeros([10 1]);
counts_model2 = zeros([10 1]);
Ncorrect_baseline2 =zeros(M,1);
Ncorrect_model2 = zeros(M,1);
for n = 1:M
    true_presence = presence_truth(:,n);

    % baseline
    scores_baseline = presence_score(:,n);
    
    [s,k] = sort(scores_baseline, 'descend');
    Nd = length(k);
    Nd = min(Nd,10);
    correct_baseline2(1:Nd) = correct_baseline2(1:Nd)+true_presence(k(1:Nd));
    counts_baseline2(1:Nd) = counts_baseline2(1:Nd)+1;
    Ncorrect_baseline2(n) = find(true_presence(k)==0, 1 )-1;
    
    % model
    scores_model = presence_score_c(:,n);
    
    [s,k] = sort(scores_model, 'descend');
    Nd = length(k); 
    Nd = min(Nd,10);
    correct_model2(1:Nd) = correct_model2(1:Nd)+true_presence(k(1:Nd));
    counts_model2(1:Nd) = counts_model2(1:Nd)+1;
    Ncorrect_model2(n) = find(true_presence(k)==0, 1 )-1;
end
perf_basline2 = correct_baseline2./counts_baseline2;
perf_model2 = correct_model2./counts_model2;

Ncorrect_bound2 = sum(presence_truth,1);

subplot(122);cla
h_t2 = hist(Ncorrect_bound2, 0:5);
ht2 = cumsum(h_t2(6:-1:2));
h_b2 = hist(Ncorrect_baseline2, 0:5);
hb2 = cumsum(h_b2(6:-1:2))./ht2;
h_m2 = hist(Ncorrect_model2, 0:5);
hm2 = cumsum(h_m2(6:-1:2))./ht2;
bar(([hb2(end:-1:1); hm2(end:-1:1)])')
title('Presence Prediction')
axis('square')
xlabel('N')
ylabel('Percentage of Images')
grid on
legend('Baseline','Context')
if doviz
    print ('../figures/topN','-dpdf');
end
%-------------------------------------------------
disp('Roc and PR curves')
ap=zeros(N,1);
ap_context=zeros(N,1);
au=zeros(N,1);
au_context=zeros(N,1);
ap_presence_context=zeros(N,1);
ap_presence=zeros(N,1);
T = linspace(0,100,11);

%------------------------------------- Square roc & pr (random)
for  c = 1:N
    fprintf('.');
    objectname = names{c};
    [recall, precision, DdetectorTest, ~ , score, correct, ap(c)] = LMrecallPrecision(test_imdb, DdetectorTest, objectname, 'nomisses');
    [recall_context, precision_context, DdetectortestUpdate, ~, score_context, correct_context, ap_context(c)] = LMrecallPrecision(test_imdb, DdetectortestUpdate, objectname, 'nomisses');
    
    [prRecall,prPrecision,foo, au(c)]= precisionRecall(presence_score(c,:), presence_truth(c,:));

    temp=0;
    for t=T 
        p=max(prPrecision(prRecall>=t));
        if isempty(p)
            p=0;
        end
        temp=temp+p/length(T);
    end
    ap_presence(c) = temp;

    [prRecall_context,prPrecision_context, foo, au_context(c)] = precisionRecall(presence_score_c(c,:), presence_truth(c,:));

    temp=0;
    for t=T 
        p=max(prPrecision_context(prRecall_context>=t));
        if isempty(p)
            p=0;
        end
        temp=temp+p/length(T);
    end
    
    ap_presence_context(c) = temp;  

    if(mod(c,20)==0)
        fprintf('%d',c);
        figure; clf
        annotation('textbox',[0.45 0.01 .1 .1],...
            'String',objectname,'FontSize',14,'FontName','Arial','EdgeColor','none');   
        subplot(221)
        plot(recall, precision, 'b', recall_context, precision_context, 'r')
        legend('Baseline','Context')
        title('Localization: precision-recall')
        axis('square')
        subplot(222)
        plot(prRecall, prPrecision, 'b', prRecall_context, prPrecision_context, 'r')
        title('Presence prediction: precision-recall')
        axis('square')
        subplot(223)
        RocCurve(score,correct,'b');
        hold on
        RocCurve(score_context,correct_context,'r');
        hold off
        title('Localization: ROC')
        subplot(224)
        RocCurve(presence_score(c,:), presence_truth(c,:), 'b');
        hold on
        RocCurve(presence_score_c(c,:), presence_truth(c,:), 'r');
        hold off
        title('Presence Prediction: ROC')
        drawnow     
    end
end
fprintf('\n');

%-------------------------------------------------- PR in line
%9 bed, 38 dish, 47 floor, 73 road, 110 water
figure(1);
set(gcf,'Position',[0 0 1000 200])
figure(2);
set(gcf,'Position',[0 0 1000 200])
i=1;
for c=[9 38 47 73 110]
    fprintf('.');
    objectname = names{c};
    [recall, precision, DdetectorTest, ~ , score, correct, ap(c)] = LMrecallPrecision(test_imdb, DdetectorTest, objectname, 'nomisses');
    [recall_context, precision_context, DdetectortestUpdate, ~, score_context, correct_context, ap_context(c)] = LMrecallPrecision(test_imdb, DdetectortestUpdate, objectname, 'nomisses');
    
    [prRecall,prPrecision,foo, au(c)]= precisionRecall(presence_score(c,:), presence_truth(c,:));

    temp=0;
    for t=T 
        p=max(prPrecision(prRecall>=t));
        if isempty(p)
            p=0;
        end
        temp=temp+p/length(T);
    end
    ap_presence(c) = temp;

    [prRecall_context,prPrecision_context, foo, au_context(c)] = precisionRecall(presence_score_c(c,:), presence_truth(c,:));

    temp=0;
    for t=T 
        p=max(prPrecision_context(prRecall_context>=t));
        if isempty(p)
            p=0;
        end
        temp=temp+p/length(T);
    end

    ap_presence_context(c) = temp;   
    if 1
        figure(1), subplot(1,5,i)
        plot(recall, precision, 'b', recall_context, precision_context, 'r','LineWidth',2)
        title(objectname)
        axis('square')
        xlabel('precision')
        ylabel('recall')
      

        figure(2), subplot(1,5,i)
        plot(prRecall, prPrecision, 'b', prRecall_context, prPrecision_context, 'r','LineWidth',2)
        title(objectname)
        axis('square')
        xlabel('precision')
        ylabel('recall')
    end
    i=i+1;    
end
figure(1)
%title('Localization: precision-recall')
if doviz
    print ('../figures/loc','-dpng');
end
figure(2)
%title('Presence prediction: precision-recall')
if doviz
    print ('../figures/pres','-dpng');
end
%--------------------------------------------------- AP listing
disp('AP per category')

disp('Localization Average precision')
fprintf('Category & baseline & context & Category & Baseline & Context & Category & Baseline & Context\\\\\n');
for c = 1:3:N
    fprintf('%s & %2.2f & %2.2f & %s & %2.2f & %2.2f & %s & %2.2f & %2.2f \\\\\n', names{c}, ap(c), ap_context(c),names{c+1}, ap(c+1), ap_context(c+1),names{c+2}, ap(c+2), ap_context(c+2));
end
fprintf('Average & %2.2f &  %2.2f\\\\\n', mean(ap), mean(ap_context));


disp('Presence Average precision')

fprintf('Category & baseline & context & Category & Baseline & Context & Category & Baseline & Context\\\\\n');
for c = 1:3:N
    fprintf('%s & %2.2f & %2.2f & %s & %2.2f & %2.2f & %s & %2.2f & %2.2f \\\\\n', names{c}, ap_presence(c),...
     ap_presence_context(c),names{c+1}, ap_presence(c+1), ap_presence_context(c+1),names{c+2}, ap_presence(c+2), ap_presence_context(c+2));
end
fprintf('Average & %2.2f &  %2.2f\\\\\n', mean(ap_presence), mean(ap_presence_context));

disp('AP improvement')
figure,
subplot(2,1,1)
bar(sort(ap_context-ap,'descend'),'r')
title('AP Improvemennt for localization')

subplot(2,1,2),
bar(sort(ap_presence_context-ap_presence,'descend'),'b')
title('AP Improvemennt for presence detection')

if doviz
    print ('../figures/improv','-dpng');
end