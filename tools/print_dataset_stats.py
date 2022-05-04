import argparse
import numpy as np

from megatron.data.gpt_dataset import get_indexed_dataset_


def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title='output data')
    group.add_argument('--dataset-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    indexed_dataset = get_indexed_dataset_(args.dataset_prefix,
                                           data_impl='mmap',
                                           skip_warmup=True)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    print(f'total_num_of_documents = {total_num_of_documents} ({total_num_of_documents:,})')

    total_num_of_tokens = np.sum(indexed_dataset.sizes)
    print(f'total_num_of_tokens = {total_num_of_tokens} ({total_num_of_tokens:,})')


if __name__ == '__main__':
    main()
