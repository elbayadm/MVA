function elt =  map(x,I)
% get ith element of array with extension on the borders
tiny = 1e-6;
elt = zeros(length(I),1);
for i = 1:length(I)
    index = I(i);
    if (index>=1) & (index<=numel(x))
        elt(i) = x(index);
    elseif index<1
            elt(i) = x(1) - tiny;
        elseif i > numel(x)
                elt(i) = x(end)+tiny;
        end
    end
end
  