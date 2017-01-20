function rgb = hsv2rgb1 (hsv)
[h, w ,~] = size(hsv);
H = hsv(:,:,1); H = H(:);
S = hsv(:,:,2); S = S(:);
V = hsv(:,:,3); V = V(:);

A = S.*cos(H);
B = S.*sin(H);
T = [1/sqrt(3) 1/sqrt(3) 1/sqrt(3); ...
     0 1/sqrt(2) -1/sqrt(2); ...
     2/sqrt(6) -1/sqrt(6) -1/sqrt(6)];
    
rgb = [V A B]*T;
R = rgb(:,1); R = reshape(R,[h w]); 
G = rgb(:,2); G = reshape(G,[h w]); 
B = rgb(:,3); B = reshape(B,[h w]); 

rgb = cat(3,R,G,B);