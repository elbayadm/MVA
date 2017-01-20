function varargout = slant_detection(im)
    % Detect edges:
    BW = im>.5; 
    % Perform the Hough transform.
    [T,R,H]  = hough_transform(BW); 
    % Find the most dominant line direction.
     tic,
     V=var(H);   
     [~, ang] = max(V);          
     varargout{1} = T(ang);
     if nargout == 2
         [~,r] = max(H(:,ang));
         varargout{2} = R(r);
     end
end

function [ T, R ,H] = hough_transform(BW) 
% Perform hough transform to detect lines in input image
    [rows, cols] = size(BW);
 
    theta_maximum = 90;
    rho_maximum = floor(sqrt(rows^2 + cols^2)) - 1;
    T = -theta_maximum:theta_maximum - 1;
    R = -rho_maximum:rho_maximum;
 
    H = zeros(length(R), length(T));
 
    
    for row = 1:rows
        for col = 1:cols
            if BW(row, col) > 0
                x = col - 1;
                y = row - 1;
                for theta = T
                    rho = round((x * cosd(theta)) + (y * sind(theta)));                   
                    rho_index = rho + rho_maximum + 1;
                    theta_index = theta + theta_maximum + 1;
                    H(rho_index, theta_index) = H(rho_index, theta_index) + 1;
                end
            end
        end
    end
     
end


