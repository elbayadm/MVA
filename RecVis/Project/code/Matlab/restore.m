addpath('data/morpho/original')
listing=importdata('test.txt');
scores =importdata('B100.txt');
%
[I,J]=ismember(listing.textdata,scores.textdata);
myscores=scores.data(J);
mynames =scores.textdata(J);
out=fopen('test_scores.txt','w');
for i = 1:length(myscores)
    fprintf(out,'%s %.2f\n',mynames{i},myscores(i));
end
fclose(out)
disp('Done')