disp('loading test detector outputs...')
if ~exist('DdetectorTest')
    load('sun09_detectorOutputs.mat','DdetectorTest');
end
M = length(test_imdb); % #Images

presence_score = zeros([N, M]);
presence_score_c = zeros([N, M]);
presence_truth = zeros([N, M]);
DdetectortestUpdate=DdetectorTest;

disp('Alternating trees')
Niters=3; % Number of iterations

fprintf('[/%d] ',floor(M/100)+1);

start_t=tic;
for n=1:M
    if(mod(n,100)==0)
        fprintf('%d..',n/100);
    end
    objects = {DdetectortestUpdate(n).annotation.object.name};
    [~,obj] = ismember(objects, names); obj = obj';
    scores = [DdetectortestUpdate(n).annotation.object.confidence]';
    
    in = find(obj>0);
    obj = obj(in);
    scores = scores(in);

    W = length(scores); 
    node_potential_g = node_potential;
   
    %Include gist features
    p_b_gist = p_b_gist_test(n,:)';
    pnp_g = node_potential(1:N,:).*[(1-p_b_gist)./(1-prob_bi1) p_b_gist./prob_bi1];
    node_potential_g(1:N,:) = pnp_g./repmat(sum(pnp_g,2),1,2);
    

    % Combine the measurement and the context model
    measurement_adjmat = sparse(N, W);
    measurement_edge_pot = sparse(2*N,2*W);
    measurement_node_pot = zeros(W,2);
    pw1lscore = zeros(W,1);
    numWindows = zeros(N,1);

    for ww=1:W   
        i = obj(ww);
        numWindows(i) = numWindows(i)+1;
        measurement_adjmat(i,ww) = 1;
        % P(c_i,ww | bi=1)
        pw1lb1 = windowScore.pKthCorrectGivenObjectPresent{i}(numWindows(i)); 
        measurement_edge_pot(2*i-1:2*i,2*ww-1:2*ww) = [1 0; 1-pw1lb1, pw1lb1];
        % p(c_i,ww|score_i,ww)
        pw1lscore(ww) = glmval(windowScore.logitCoef{i}, scores(ww), 'logit');
        pCorrect = pw1lb1*prob_bi1(i);
        measurement_node_pot(ww,:) = [(1-pw1lscore(ww))/(1-pCorrect) pw1lscore(ww)/pCorrect];
    end

    for i=1:N 
        node_potential_g(i,2) = node_potential_g(i,2)*prod(1-windowScore.pKthCorrectGivenObjectPresent{i}(numWindows(i)+1:end));
    end
        
    adjmat = [A measurement_adjmat; measurement_adjmat' sparse(W,W)];
    node_pot = [node_potential_g; measurement_node_pot];
    edge_pot = [edge_potential measurement_edge_pot; measurement_edge_pot' sparse(2*W,2*W)];
    node_pot = node_pot./repmat(sum(node_pot,2),1,2);

    
    image_size(1) = DdetectortestUpdate(n).annotation.imagesize.ncols;
    image_size(2) = DdetectortestUpdate(n).annotation.imagesize.nrows;   
    [loc_index,loc_measurements,~] = getWindowLoc(DdetectortestUpdate(n).annotation.object(in),names,image_size,heights);

    %Save initial values
    init_node_pot = node_pot;
    init_edge_pot = edge_pot;   

    %Messagge passing order------------------------------------------
        crntN=length(adjmat);
        tree_msg = zeros(2*(crntN-1),2);
        tree_msgIndex = crntN;
        prevNodes = [];
        currentNodes = root;
        while (tree_msgIndex <= 2*(crntN-1))
          allNextNodes = [];
          for (i = 1:length(currentNodes))
            nextNodes = setdiff(find(adjmat(currentNodes(i),:)),prevNodes);
            Nnext = length(nextNodes);
            tree_msg(tree_msgIndex:tree_msgIndex+Nnext-1,:) = ...
              [repmat(currentNodes(i),Nnext,1), nextNodes'];
            tree_msgIndex = tree_msgIndex + Nnext;
            allNextNodes = [allNextNodes, nextNodes];
          end

          prevNodes = [prevNodes, currentNodes];
          currentNodes = allNextNodes;
        end
        tree_msg(1:crntN-1,:) = fliplr(flipud(tree_msg(crntN:end,:)));
    %----------------------------------------------------------------
    for ite=1:Niters
        %------------------------------------------ p(b,c|g,s)

        MAP = viterbi(adjmat, node_pot, edge_pot, tree_msg);   
        correct_detection = (MAP(N+1:end)==2); %(bi=1)

        %------------------------------------------ update potentials L|b
        
        InfLocation = sparse(2*N,2*N);
        potLocation = zeros(2*N,1);
        ii = 2*(root-1)+1:2*root;
        InfLocation(ii,ii) = diag(locationPot.Imargin(root,:));
        potLocation(ii) = locationPot.hmargin(root,:)';
        for e=1:size(edges,1)
            p = edges(e,1);
            c = edges(e,2);
            ii = 2*(c-1)+1:2*c;
            if(MAP(c)==0)
                InfLocation(ii,ii) = InfLocation(ii,ii)+diag(locationPot.Imargin(c,:));
                potLocation(ii) = potLocation(ii)+locationPot.hmargin(c,:)';       
            elseif(MAP(p)==0)
                ii = K*(c-1)+1:K*c;
                InfLocation(ii,ii) = InfLocation(ii,ii)+diag(locationPot.Icond_np(c,:));
                potLocation(ii) = potLocation(ii)+locationPot.hcond_np(c,:)';
            else 
                Icc = locationPot.Icond(e,1:2);
                Ipc = locationPot.Icond(e,3:4);
                hc = locationPot.hcond(e,:);
                Ipair = [diag(Ipc.^2./Icc), diag(Ipc); diag(Ipc), diag(Icc)];
                hpair = [Ipc.*hc./Icc, hc]';
                ii_pc = [2*(p-1)+1:2*p,2*(c-1)+1:2*c];
                InfLocation(ii_pc,ii_pc) = InfLocation(ii_pc,ii_pc) + Ipair;
                potLocation(ii_pc) = potLocation(ii_pc) + hpair;       
            end
        end
        
        
        %----------------------------------- update potentials W|b

        sub_Loc=loc_measurements(correct_detection,:);
        obj_index=loc_index(correct_detection);
        N = size(detectionWindowLoc.meanCorrect,1);
        NW = length(obj_index);
        C = sparse(2*NW,2*N);
        measurement_variance = zeros(2*NW,1);
        for i=1:NW
            obj = obj_index(i);
            ii = 2*(i-1)+1:2*i;
            obj2 = 2*(obj-1)+1:2*obj;
            C(ii,obj2) = speye(2);
            sub_Loc(i,:) = sub_Loc(i,:) - detectionWindowLoc.meanCorrect(obj,:);
            measurement_variance(ii) = detectionWindowLoc.varianceCorrect(obj,:);
        end

        Rinv = spdiags(1./measurement_variance,0,2*NW,2*NW);
        InfMeasurement = C'*Rinv*C;
        sub_Loc = sub_Loc';
        potMeasurement = sparse(C'*Rinv*sub_Loc(:));
        %----------------------------------------------------- Combine L,W

        MAP_spatial = (InfLocation + InfMeasurement)\(potLocation + potMeasurement);
        MAP_spatial = reshape(MAP_spatial',2,N)'; 
    
        % Update binary potentials using location estimates
        % ----------------------------------Compute p(cik | bi, Li, Wik)

        node_pot=init_node_pot;
        relativeLoc = loc_measurements - MAP_spatial(loc_index,:);

        std = 2*sqrt(detectionWindowLoc.varianceCorrect);
        prodStd = prod(std,2);
        w2 = normpdf(relativeLoc(:,1),detectionWindowLoc.meanCorrect(loc_index,1),std(loc_index,1))...
            .*normpdf(relativeLoc(:,2),detectionWindowLoc.meanCorrect(loc_index,2),std(loc_index,2));

        w1 = normpdf(1,0,1)^2*ones(length(loc_index),1)./prodStd(loc_index);

        temp = node_pot(N+1:end,:).*[w1 w2];
        node_pot(N+1:end,:) = temp./repmat(sum(temp,2),1,2);

        % --------------------------------------------------Compute p(bj | bi, Li, Lj)    
        edge_pot=init_edge_pot;
        for e=1:N-1
            p = edges(e,1);
            c = edges(e,2);
            
            w_1 = gaussian_stats(MAP_spatial(c,:),locationPot.Imargin(c,:),locationPot.hmargin(c,:)); % P(bj = 0 | bi, Li, Lj)
            w12 = gaussian_stats(MAP_spatial(c,:),locationPot.Icond_np(c,:),locationPot.hcond_np(c,:)); % P(bj = 1 | bi=0, Li, Lj)
            
            Icond = locationPot.Icond(e,1:2);
            hcond = locationPot.hcond(e,:) - locationPot.Icond(e,3:4).*MAP_spatial(p,:);
            w22 = gaussian_stats(MAP_spatial(c,:),Icond,hcond); % P(bj=1 | bi=1, Li, Lj)

            temp = edge_pot(2*p-1:2*p,2*c-1:2*c).*[w_1, w12; w_1, w22]/(w_1*2+w12+w22);
            edge_pot(2*p-1:2*p,2*c-1:2*c) = temp ./repmat(sum(temp(:)),2,2);
            edge_pot(2*c-1:2*c,2*p-1:2*p) = edge_pot(2*p-1:2*p,2*c-1:2*c)';    
        end                
    end
    
    node_marginals = sum_product_binary(adjmat, node_pot, edge_pot, tree_msg);
    new_scores = node_marginals(:,2);   
    
    % Insert scores in final result struct
    tiny = 1e-5;  % Used to avoid numerical issues.
    for i = 1:W
        DdetectortestUpdate(n).annotation.object(in(i)).confidence = (1-tiny)*new_scores(i+N)+tiny*(scores(i)+10)/20;
        DdetectortestUpdate(n).annotation.object(in(i)).p_w_s =  (1-tiny)*pw1lscore(i)+tiny*(scores(i)+10)/20;
    end      

    % Insert presence prediction
    [foo, trueobj] = ismember({test_imdb(n).annotation.object.name}, names);
    presence_truth(setdiff(trueobj,0),n) = 1;        
    for m = 1:N
        % For the baseline model p(bi=1)=the most confident detection of category i
         presence_score(m,n) = max([0; pw1lscore(obj==m)]);
    end         
    presence_score_c(:,n) = (1-tiny)*new_scores(1:N)+tiny*presence_score(:,n);
end
alter=toc(start_t);
fprintf('\nProcessing time per image %.3f\n',alter/M);
fprintf('\nDone.\n');  
