function Y = myloss(X,c,loss_type,dzdy)
c = reshape(c, 1,1,1,[]);
if nargin <= 3 || isempty(dzdy) % forward step - compute loss
    switch loss_type
        case 'squareloss'
            %fprintf('size(X): %d %d %d %d\n',size(X))
            %fprintf('size(c): %d %d %d %d\n',size(c))
            Y = .5*(X-c).^2;
        case 'logloss'
            %fprintf('size(X): %d %d %d %d\n',size(X))
            %fprintf('size(c): %d %d %d %d\n',size(c))
            s = 2*c-1;
            Y = log(1+exp(-s.*X));
        case 'hingeloss'
            %fprintf('size(X): %d %d %d %d\n',size(X))
            %fprintf('size(c): %d %d %d %d\n',size(c))
            Y = max(0, 1 - c.*X) ;
        otherwise
            error('Loss type not supported')
    end
    Y = sum(Y(:));
else % backward step - compute loss derivative
    switch loss_type
        case 'squareloss'
            Y = dzdy * (X-c);
            %fprintf('size(Y): %d %d %d %d\n',size(Y))
        case 'logloss'
            s = 2*c-1;
            Y = -s.*dzdy.*exp(-s.*X)./(1+exp(-s.*X));
        case 'hingeloss'
            Y = - dzdy.*c.* (c.*X < 1);
        otherwise
            error('Loss type not supported');
    end
end
