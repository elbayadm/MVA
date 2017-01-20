function hsv = rgb2hsv1 (rgb)
[h, w ,~] = size(rgb);
R = rgb(:,:,1); R = R(:);
G = rgb(:,:,2); G = G(:);
B = rgb(:,:,3); B = B(:);
T = [1/sqrt(3) 1/sqrt(3) 1/sqrt(3); ...
     0 1/sqrt(2) -1/sqrt(2); ...
     2/sqrt(6) -1/sqrt(6) -1/sqrt(6)];
    
vab = [R G B]*T';
V = vab(:,1);                         V = reshape(V, [h w]);
S = sqrt(vab(:,2).^2 + vab(:,3).^2);  S = reshape(S, [h w]);
H = atan2(vab(:,3),vab(:,2));         H = reshape(H, [h w]);
hsv = cat(3,H,S,V);