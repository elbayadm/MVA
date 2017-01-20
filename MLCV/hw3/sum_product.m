function marginals=sum_product(vertex,Neighbors,F)
    card = [3 2 4 2 2]; % cardinality of each r.v
    marginals = ones(card(vertex),1);
    for adj=Neighbors{vertex} % loop over adjacent vertices
        marginals = marginals.*sum_mu(adj,vertex,Neighbors,F,card);
    end
end

function m=sum_mu(j,i,Neighbors,F,card)
    N = setdiff(Neighbors{j},i);
    if isempty(N)
        m = sum(F{i}{j},2);
    else
        prod = ones(card(j),1);
        for p=N
            prod = prod.*sum_mu(p,j,Neighbors,F,card);
        end
        m=F{i}{j}*prod;
    end
end

