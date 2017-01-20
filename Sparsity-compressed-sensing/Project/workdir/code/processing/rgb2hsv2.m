function hsv = rgb2hsv2 (rgb)
tiny = 1e-20;
R = rgb(:,:,1); 
G = rgb(:,:,2);
B = rgb(:,:,3); 

m = min(rgb,[],3); M = max(rgb,[],3);
delta = M - m;
S = delta ./ (M + tiny);
V = M;


H = (R==M).*(G-B)./(delta+tiny)+...
    (G==M).*(2+(B-R)./(delta+tiny))+...
    (B==M).*(4+(R-G)./(delta+tiny));
    
H(H<0)= H(H<0)+6;
H = H /6;
hsv = cat(3,H,S,V);
