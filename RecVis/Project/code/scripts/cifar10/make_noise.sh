

cd $(dirname ${BASH_SOURCE[0]})/../../

IN=data/cifar10-py
OUT=data/cifar10

NOISE_LEVEL=0.8
SIZE=50000

echo "Generating noise with level" $NOISE_LEVEL "database size:" $SIZE
# make auxiliary files
echo "Making auxiliary files"
python py/generate_noisy_labels.py ${OUT} --level ${NOISE_LEVEL}  --size ${SIZE}
python py/convert_to_blobproto.py \
    ${OUT}/matrix_q${NOISE_LEVEL}.pkl ${OUT}/true_matrix_q${NOISE_LEVEL}.binaryproto

# make databases
make_imageset() {
    caffe/build/tools/convert_imageset \
        -backend lmdb -resize_height 32 -resize_width 32 \
        ${OUT}/ ${OUT}/$1.txt ${OUT}/$1
}

make_labelset() {
    caffe/build/tools/convert_labelset \
        -backend lmdb ${OUT}/$1.txt ${OUT}/$1
}

make_labelset 'labels_noisy_'${NOISE_LEVEL}

