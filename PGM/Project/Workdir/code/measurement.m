disp('loading detector outputs...')
load('sun09_detectorOutputs.mat','DdetectorTraining','logitCoef','validcategories','MaxNumDetections');

%----------------
% P(c|b) & P(c|s)
%----------------

% Learn p(c_{ik} | b) = probability of correct detection when object is present/not present
% Learn p(c | s) = probability of correct detection based on the detector score (logistic function)
%   Provided with the database.

disp('Training window scores and probabilities of correct detections')

countsDetections = zeros(N,1);
countsCorrect = zeros(N,1);
countsImgTrueObj = zeros(N,1);
countsDetectionsInImgTrueObj = zeros(N,1);
countsCorrectInImgTrueObj = zeros(N,1);
countsKthCorrect = cell(N,1);
maxDetections = MaxNumDetections*ones(N,1);
for i=1:N
    countsKthCorrect{i} = zeros(1,maxDetections(i));
end
fprintf('[/%d] ',floor(M/100)+1);
for n=1:M    
    if(mod(n,200)==0)
        fprintf('%d..',n/100);
    end
    objects = {train_imdb(n).annotation.object.name};
    existObj = ismember(names,objects);
    countsImgTrueObj(existObj)  = countsImgTrueObj(existObj) + 1;
    
    objects = {DdetectorTraining(n).annotation.object.name};
    detections = {DdetectorTraining(n).annotation.object.detection};
    [foo,obj] = ismember(objects, names); obj = obj'; 
    [foo,correct] = ismember(detections,'correct'); 
    
    valid = find(obj>0);
    obj = obj(valid);
    correct = correct(valid);
    unique_obj = unique(obj);
    
    for oi=1:length(unique_obj);
        o = unique_obj(oi);
        windows = find(obj==o);
        numDetections = length(windows);
        % if(numDetections > maxDetections(o))
        %     fprintf('  Warning: the maximum number of candidate windows is %d, which is larger than the expected value %d\n',numDetections,maxDetections(o));
        %     countsKthCorrect{o} = [countsKthCorrect{o} zeros(1,numDetections-maxDetections(o))];
        %     maxDetections(o) = numDetections;
        % end
        countsDetections(o) = countsDetections(o)+numDetections;
        correctDetections = correct(windows);
        countsKthCorrect{o}(1:numDetections) = countsKthCorrect{o}(1:numDetections) + correctDetections;
        countsCorrect(o) = countsCorrect(o) + sum(correctDetections);
        if(existObj(o))
            countsCorrectInImgTrueObj(o) = countsCorrectInImgTrueObj(o) + sum(correctDetections);
            countsDetectionsInImgTrueObj(o) = countsDetectionsInImgTrueObj(o) + numDetections;
        end
    end
end
fprintf('\n');
clear windowScore

windowScore.name = names;
windowScore.countsCorrectDetections = countsCorrect;
windowScore.countsFalseAlarms = countsDetections - countsCorrect;
windowScore.maxCandWindows = maxDetections;

windowScore.pCorrect = countsCorrect ./ countsDetections;
windowScore.pCorrectGivenObjectPresent = countsCorrectInImgTrueObj./ countsDetectionsInImgTrueObj;

for i=1:N
    windowScore.pKthCorrectGivenObjectPresent{i} = (countsKthCorrect{i}+1)/(countsImgTrueObj(i)+maxDetections(i));
    [flag,j] = ismember(names{i},validcategories);
    if(flag)
        windowScore.logitCoef{i} = logitCoef{j};
    else
        ferror('Cannot find the logistic coefficients for %s\n',names{i});
    end 
end


% Integrate L to c.

diffSqCorrect = zeros(N,2);
diffSqFalse = zeros(N,2);
diffCorrect = zeros(N,2);
diffFalse = zeros(N,2);
Ncorrect = zeros(N,1);
Nfalse = zeros(N,1);
fprintf('[/%d] ',floor(M/100)+1);
for m=1:M
    if(mod(m,200)==0)
        fprintf('%d..',m/100);
    end
    [foo,true_obj] = ismember({train_imdb(m).annotation.object.name},names); 
    unique_true_obj = setdiff(unique(true_obj'),0);
    image_size(1) = train_imdb(m).annotation.imagesize.ncols;
    image_size(2) = train_imdb(m).annotation.imagesize.nrows;                
    [true_loc_index,true_loc] = getWindowLoc(train_imdb(m).annotation.object,names,image_size,heights);
    
    [~, obj] = ismember({DdetectorTraining(m).annotation.object.name},names(unique_true_obj));
    valid = (obj>0);
    [loc_index,loc_measurements] = getWindowLoc(DdetectorTraining(m).annotation.object(valid),names,image_size,heights);        
    isCorrect = ismember({DdetectorTraining(m).annotation.object(valid).detection}, {'correct'});

    for o=1:length(unique_true_obj)
        n = unique_true_obj(o);
        obj_n = (loc_index==n);
        med_true_loc = median(true_loc(true_loc_index==n,:),1);
            
        isCorrect_n = isCorrect(obj_n);
        Ncorrect(n) = Ncorrect(n) + sum(isCorrect_n);
        Nfalse(n) = Nfalse(n) + sum(~isCorrect_n);
        
        relativeWindowLoc = loc_measurements(obj_n,:) - repmat(med_true_loc,sum(obj_n),1);
        diffSqCorrect(n,:) = diffSqCorrect(n,:) + sum(relativeWindowLoc(isCorrect_n,:).^2,1);
        diffSqFalse(n,:) = diffSqFalse(n,:) + sum(relativeWindowLoc(~isCorrect_n,:).^2,1);
        diffCorrect(n,:) = diffCorrect(n,:) + sum(relativeWindowLoc(isCorrect_n,:),1);
        diffFalse(n,:) = diffFalse(n,:) + sum(relativeWindowLoc(~isCorrect_n,:),1);
    end
end
fprintf('\n');

detectionWindowLoc.Ncorrect = Ncorrect;
detectionWindowLoc.Nfalse = Nfalse;
detectionWindowLoc.meanCorrect = diffCorrect./repmat(Ncorrect,1,2);
detectionWindowLoc.meanFalse = diffFalse./repmat(Nfalse,1,2);
detectionWindowLoc.varianceCorrect = diffSqCorrect./repmat(Ncorrect,1,2)-detectionWindowLoc.meanCorrect.^2;
detectionWindowLoc.varianceFalse = diffSqFalse./repmat(Nfalse,1,2)-detectionWindowLoc.meanFalse.^2;



