

cd $(dirname ${BASH_SOURCE[0]})/../../

IN=data/Listings
OUT=data/morpho
IM=data/morpho/glasses/

NOISE_LEVEL=0.02
SIZE=4152
echo "Generating noise with level" $NOISE_LEVEL "database size:" $SIZE
# make auxiliary files
echo "Making auxiliary files"
python py/generate_asym_noisy_labels.py ${OUT} --level ${NOISE_LEVEL} --size ${SIZE}
# python py/convert_to_blobproto.py \
#     ${OUT}/matrix_qb${NOISE_LEVEL}.pkl ${OUT}/matrix_qb${NOISE_LEVEL}.binaryproto

# make databases
make_imageset() {
    caffe/build/tools/convert_imageset \
    	${IM} \
        ${OUT}/$1.txt\
        ${OUT}/$1
}

make_labelset() {
    caffe/build/tools/convert_labelset \
        -backend lmdb ${OUT}/$1.txt ${OUT}/$1
}

# make_labelset noisyb0.15_labels
#make_imageset clean2
#make_labelset noisy${NOISE_LEVEL}_labels

# make_imageset 'noisy_train'${NOISE_LEVEL}'_'${SIZE}
# rm -rf 'clean_train'${SIZE}
# make_imageset 'clean_train'${SIZE}

# caffe/build/tools/compute_image_mean \
#     -backend lmdb \
#     ${OUT}/noisy_train${NOISE_LEVEL}_${SIZE} ${OUT}/cifar10_mean_${SIZE}.binaryproto
