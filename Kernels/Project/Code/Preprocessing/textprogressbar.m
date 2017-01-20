function textprogressbar(c,varargin)
% Adapted fom : http://blogs.mathworks.com/loren/2007/08/01/monitoring-progress-of-a-calculation/
%-------------------------------------------------------------------------------
% Initialization
persistent strCR;      

% Vizualization parameters
strDotsMaximum  = 10; 
if isempty(strCR) && ~ischar(c),
    % Progress bar must be initialized with a string
    error('The text progress must be initialized with a string');
elseif isempty(strCR) && ischar(c),
    % Progress bar - initialization
    fprintf('%s',c);
    strCR = -1;
elseif ~isempty(strCR) && ischar(c),
    % Progress bar  - termination
    clear strCR
    fprintf([' ' c '\n']);
elseif isnumeric(c)
    n = varargin{1};
    % Progress bar - normal progress
    c = floor(c/n*100);
    percentageOut = [num2str(c) '%% '];
    nDots = floor(c/strDotsMaximum);
    dotOut = ['[' repmat('.',1,nDots) repmat(' ',1,strDotsMaximum-nDots) ']'];
    strOut = [percentageOut dotOut];
    
    % Print it on the screen
    if strCR == -1,
        % Don't do carriage return during first run
        fprintf(strOut);
    else
        % Do it during all the other runs
        fprintf([strCR strOut]);
    end
    
    % Update carriage return
    strCR = repmat('\b',1,length(strOut)-1);
    
else
    % Any other unexpected input
    error('Unsupported argument type');
end
