function marginals=max_product(vertex,Neighbors,F)
    card = [3 2 4 2 2]; % cardinality of each r.v
    marginals = ones(card(vertex),1);
    for adj=Neighbors{vertex}
        marginals = marginals.*max_mu(adj,vertex,Neighbors,F,card);
    end
end


function m=max_mu(j,i,Neighbors, F, card)
    N = setdiff(Neighbors{j},i);
    if isempty(N)
        m=max(F{i}{j},[],2);
    else
        prod = ones(card(j),1);
        for p=N
            prod = prod.*max_mu(p,j,Neighbors,F,card);
        end
        m=transpose(max(F{j}{i}.*repmat(prod,[1 card(i)])));
    end
end