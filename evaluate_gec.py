from evaluators.postprocess import PostProcess

import argparse
from utils.log import setup_log
import logging

logger = logging.getLogger(__name__)

def parse_args():
    
    parser = argparse.ArgumentParser(description="Argument parser for the script.")

    parser.add_argument('--dir', type=str, required=True, help='Directory path')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--device', type=int, default=0, help='Device number')
    parser.add_argument('--pre_split_length_for_infer', type=int, default=None, help='Pre-split length for inference')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    setup_log(args.dir)
    processor = PostProcess(args=args)
    processor.post_process_and_save()
