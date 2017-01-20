function im_ = im_trim(im)
    th = mean(mean(im))/2;
    vp = mean(im, 1) > th;
    hp = mean(im, 2) > th;
    fc = max(find(vp, 1, 'first')-1,1);
    lc = min(find(vp, 1, 'last')+1,length(vp));
    fr = max(find(hp, 1, 'first')-1,1);
    lr = min(find(hp, 1, 'last')+1,length(hp));
    im_ = im(fr:lr,fc:lc);
 end