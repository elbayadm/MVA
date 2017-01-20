%% The context model

disp('Learning the spatial prior...') 

%----------------
% P(L|b)
%----------------

image_size = zeros(M,2);
% Median values for redundant objects:
med = zeros(M,N,2);
for m=1:M
    % List of objects in the mth image:
    objects = {train_imdb(m).annotation.object.name};
    % Retrieve the indices
    [~,obj] = ismember(objects, names); 
    obj = obj'; 
    in= find(obj>0); 
    obj = obj(in);
    unique_obj = unique(obj);    
    % Get image size:
    image_size(m,1) = train_imdb(m).annotation.imagesize.ncols;
    image_size(m,2) = train_imdb(m).annotation.imagesize.nrows; 

    % loc_measurements=  location parameters (Ly,log(Lz))
    [~,loc_measurements,~] = getWindowLoc(train_imdb(m).annotation.object(in),names,image_size(m,:),heights);

    for oi=1:length(unique_obj)
        o = unique_obj(oi);
        med(m,o,:) = median(loc_measurements(obj==o,:),1); 
    end  
end

disp('Estimating parameters for the gaussian spatial distributions...') 

mu_Lij = zeros(N,N*4);
%mu_Lij(i,[4*(j-1)+1:4*j]) = mean of (Li, Lj) for i < j | (bi=1,bj=1)
sigma_Lij = zeros(N*4,N*4);
%sigma_Lij([4*(i-1)+1:4*i],[4*(j-1)+1:4*j]) = covariance of (Li,Lj) | (bi=1,bj=1)
mu_Lij0 = zeros(N,N*2);
%mu_Lij0(i,[2*(j-1)+1:2*j]) = mean of Li  | (bi=1,bj=0)
sigma_Lij0 = zeros(N,N*2);
%sigma_Lij0([2*(i-1)+1:2*i],[2*(j-1)+1:2*j]) = covariance of Li | (bi=1,bj=0)
temp=logical(b);
for i=1:N
    ind_i = 2*(i-1)+1:2*i;
    bi = temp(i,:);
    Li = squeeze(med(bi,i,:));

    mean_Li = mean(Li,1);
    med_Li = median(Li,1);
    mu_Lij0(i,ind_i) = med_Li;
    sigma_Lij0(i,ind_i) = mean(Li.^2,1) - mean_Li.^2;    
    for j=1:N
        if(j==i)
            continue;
        end
        ind_j = 2*(j-1)+1:2*j;
        
        m_bij0 = temp(i,:) & ~temp(j,:);
        Lij0 = squeeze(med(m_bij0,i,:));
        
        mean_Lij0 = mean(Lij0,1);
        med_Lij0 = median(Lij0,1);
        mu_Lij0(i,ind_j) = med_Lij0;
        sigma_Lij0(i,ind_j) = mean(Lij0.^2,1) - mean_Lij0.^2;
        
        if (j>i)
            ind2_i = 4*(i-1)+1:4*i;
            ind2_j = 4*(j-1)+1:4*j;
            m_bij = temp(i,:) & temp(j,:);
            Li = squeeze(med(m_bij,i,:));
            xj = squeeze(med(m_bij,j,:));
            if(size(Li,2)==1)
                Li = Li'; xj = xj';
            elseif(isempty(Li))
                continue;
            end
            mean_Li = mean(Li,1);
            mean_xj = mean(xj,1);
            med_Lij = median([Li, xj],1);
            mu_Lij(i,ind2_j) = med_Lij;
            Li2 = Li - repmat(mean_Li,size(Li,1),1);
            Lj2 = xj - repmat(mean_xj,size(xj,1),1);
            cov_ij = mean(Li2.*Lj2,1);
            cov_i = mean(Li2.^2,1);
            cov_j = mean(Lj2.^2,1);
            sigma_Lij(ind2_i,ind2_j) = [diag(cov_i),diag(cov_ij);diag(cov_ij),diag(cov_j)];
        end
    end
end
clear temp
edges=tree_msg(N:end,:);
%----------------------------------------- Evaluate nodes and edges potentials:
edge_index = sparse(N,N);
% -------------------------Marginal
var_margin = zeros(N,2); 
mean_margin = zeros(N,2);
%--------------------------When b_parent=0
var_cond_np = zeros(N,2);  
mean_cond_np = zeros(N,2); 

root = edges(1,1);
var_margin(root,:) = sigma_Lij0(root,2*(root-1)+1:2*root);
mean_margin(root,:) = mu_Lij0(root,2*(root-1)+1:2*root);
var_cond_np(root,:) = var_margin(root,:);
mean_cond_np(root,:) = mean_margin(root,:);

for e=1:N-1
    p = edges(e,1); % parent
    c = edges(e,2); % child
    edge_index(p,c) = e;
    
    ind_c = 2*(c-1)+1:2*c;
    var_margin(c,:) = sigma_Lij0(c,ind_c);
    mean_margin(c,:) = mu_Lij0(c,ind_c);    
    
    ind_p = 2*(p-1)+1:2*p;
    var_cond_np(c,:) = sigma_Lij0(c,ind_p);
    mean_cond_np(c,:) = mu_Lij0(c,ind_p);
    
    if(sum(var_cond_np(c,:)) <=0 || isnan(var_cond_np(c,2))) % if always co-occuring    
        var_cond_np(c,:) = var_margin(c,:);
        mean_cond_np(c,:) = mean_margin(c,:);
    end
end

locationPot.Imargin = 1./var_margin; % Imargin(Lc,:) is Information matrix of p(Lc | bc = 1)
locationPot.hmargin = locationPot.Imargin.*mean_margin;
locationPot.Icond_np = 1./var_cond_np;
locationPot.hcond_np = locationPot.Icond_np.*mean_cond_np;

Icond = zeros(N-1,4);
hcond = zeros(N-1,2); % potential h of Lc|Lp

for e=1:N-1
    p = edges(e,1);
    c = edges(e,2);
    
    ind2_p = 4*(p-1)+1:4*p;
    ind2_c = 4*(c-1)+1:4*c;
    
    if (p < c)
        cov_pair = sigma_Lij(ind2_p,ind2_c).*spdiags(ones(4,3),[-2,0,2],4,4);
        mu_pair = mu_Lij(p,ind2_c);
    else
        mu_pair = mu_Lij(c,ind2_p);
        mu_pair = [mu_pair(3:end), mu_pair(1:2)];
        cov_pair = sigma_Lij(ind2_c,ind2_p).*spdiags(ones(4,3),[-2,0,2],4,4);
        cov_pair = [cov_pair(3:end,3:end), cov_pair(1:2,3:end); cov_pair(1:2,3:end), cov_pair(1:2,1:2)];
    end

    % Fix if non positive-definite covariance matrix
    if(prod(diag(cov_pair))<=0 || min(eig(full(cov_pair)))<=1e-10)
        cov_pair = diag([sigma_Lij0(p,2*(p-1)+1:2*p),sigma_Lij0(c,2*(c-1)+1:2*c)]);
        mu_pair = [mu_Lij0(p,2*(p-1)+1:2*p),mu_Lij0(c,2*(c-1)+1:2*c)];
    end
    %---------------------------------- Information of Lc|Lp
    Ipc = inv(cov_pair);
    Icond(e,1:2) = diag(Ipc(3:end,3:end));  
    Icond(e,3:end) = diag(Ipc(1:2,3:4)); 
    hpc = mu_pair*Ipc;
    hcond(e,:) = hpc(3:end);    
end

edge_index = edge_index + edge_index';    
locationPot.Icond = Icond; 
locationPot.hcond = hcond; 
