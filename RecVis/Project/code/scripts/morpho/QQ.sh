cd $(dirname ${BASH_SOURCE[0]})/../../
IN=data/infogain
caffe=/usr/local/cellar/caffe

# python2 py/Q.py --p00 .9879 --p11 .8375 --th th${th}
# echo 'Converting'
python py/convert_to_blobproto.py $IN'/Q15.pkl' $IN/'Q15.binaryproto'

make_imageset() {
    ${caffe}/build/tools/convert_imageset \
    -backend lmdb \
    	${IN}/../glasses/  \
        ${IN}/$1.txt \
        ${IN}/$1
}

make_labelset() {
    ${caffe}/build/tools/convert_labelset \
    -backend lmdb ${IN}/$1.txt ${IN}/$1
}

# make_imageset images_${th}

# make_labelset balanced_labels_${th}
# make_labelset bstrap6_balanced_labels_30_30
# make_labelset strap1_labels_30_30
# make_labelset cleaned30
# make_imageset images_crop_blur
# make_imageset test_lmdb_crop_blur

