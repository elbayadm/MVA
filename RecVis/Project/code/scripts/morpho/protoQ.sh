cd $(dirname ${BASH_SOURCE[0]})/../../

OUT=data/morpho
NOISE_LEVEL=b0.1

python py/convert_to_blobproto.py \
    ${OUT}/matrix_q${NOISE_LEVEL}.pkl ${OUT}/matrix_q${NOISE_LEVEL}.binaryproto