
cd $(dirname ${BASH_SOURCE[0]})/../../

IN=data/cifar10-py
OUT=data/cifar10
LMDB=data/cifar10-lmdb

# setup output directory
#rm -rf ${OUT}
mkdir -p ${OUT}

echo "Creating LMDB..."
python py/raw.py ${IN} ${OUT}
echo "Done."


# make auxiliary files
echo "Making auxiliary files"
python py/convert_to_blobproto.py \
    ${OUT}/identity.pkl ${OUT}/identity.binaryproto

# make databases
make_imageset() {
    caffe/build/tools/convert_imageset \
        -backend lmdb -resize_height 32 -resize_width 32 \
        ${OUT}/ ${OUT}/$1.txt ${LMDB}/cifar10_$1_lmdb
}


make_imageset 'test'
make_imageset 'train'

caffe/build/tools/compute_image_mean \
    -backend lmdb \
    ${LMDB}/cifar10_train_lmdb ${LMDB}/mean.binaryproto