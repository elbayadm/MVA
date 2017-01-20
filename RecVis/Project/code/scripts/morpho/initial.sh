

cd $(dirname ${BASH_SOURCE[0]})/../../

OUT=data/morpho

# setup output directory
#rm -rf ${OUT}
mkdir -p ${OUT}

# make auxiliary files
echo "Making auxiliary files"
# python py/identity.py ${OUT}
# python py/convert_to_blobproto.py \
#     ${OUT}/identity.pkl ${OUT}/identity.binaryproto


# Use pre-built lmdbs

caffe/build/tools/compute_image_mean \
    -backend lmdb \
    ${OUT}/clean ${OUT}/mean.binaryproto