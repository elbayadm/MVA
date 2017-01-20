import os.path as osp
import numpy as np
from argparse import ArgumentParser
from utils import pickle

def write_matrix(mat, file_path):
    content = [' '.join(map(str, r)) for r in mat]
    with open(file_path, 'w') as f:
        f.write('\n'.join(content))

def main(args):
    mkdir_if_missing(args.output_dir)
    matrix_I = np.identity(2)
    write_matrix(matrix_I, osp.join(args.output_dir, 'identity.txt'))
    pickle(matrix_I, osp.join(args.output_dir, 'identity.pkl'))

if __name__ == '__main__':
    parser = ArgumentParser(
        description="2x2 identity matrix")
    parser.add_argument('output_dir')
    args = parser.parse_args()
    main(args)