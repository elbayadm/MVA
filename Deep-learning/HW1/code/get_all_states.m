function states = get_all_states(nnodes,V)
T1 = (0:numel(V)^nnodes-1)+(1/2) ;
T2 = numel(V).^(1-nnodes:0);
IND = rem(floor((T1(:) * T2(:)')),numel(V)) + 1 ;
states = V(IND) ;    