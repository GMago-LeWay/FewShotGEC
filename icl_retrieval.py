import argparse
import os
import json
import logging
import random
from tqdm import tqdm
from transformers import pipeline
import torch
from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser
from configs.data_arguments import DataConfig
from configs.icl_arguments import ICLConfig
from trl import (
    ModelConfig,
    SFTConfig,
    RichProgressCallback,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset, Features, Value, Split

from data.icl_dataset import ICLDataset

from data.group_dataset import GroupDataset, InstructionDataset
from data.check_jsonl import extract_matching_valid_lines, rewrite_jsonl_with_valid_lines
from utils.log import setup_log, log_config
from utils.random import init_seed


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = TrlParser((ModelConfig, DataConfig, ICLConfig))
    model_config, data_config, icl_config = parser.parse_args_and_config()

    init_seed(icl_config.seed)

    setup_log(icl_config.output_dir)

    logger.info(f"Model arguments:")
    log_config(model_config, logger)
    logger.info(f"Data arguments:")
    log_config(data_config, logger)
    logger.info(f"In-context learning arguments:")
    log_config(icl_config, logger)

    # get dataset of ICL instructions
    dataset_controller = ICLDataset(data_args=data_config, model_args=model_config, icl_args=icl_config)

    # judge if we need to retrieve by external keys (mode default, random, text does not need key_text_dir from external source)
    mode_do_not_need_key = {"default", "random", "text", "bm25"}
    example_modes = {icl_config.in_domain_example_mode}
    example_modes.add(icl_config.cross_domain_example_mode)
    # judge if example modes set are all included in mode that do not need key
    if example_modes.issubset(mode_do_not_need_key):
        external_key = False
        logger.info(f"No need to retrieve by external keys.")
    else:
        external_key = True
        logger.info(f"Need to retrieve by external keys. From {icl_config.key_text_dir}")

    # infer for every dataset
    for dataset in dataset_controller.get_raw_datasets():
        data = dataset['dataset']

        logger.info(f"ICL retrieval on dataset {dataset['dataset_name']}")

        # load keys
        if external_key:
            key_jsonl = os.path.join(icl_config.key_text_dir, dataset['dataset_name'], 'predictions.jsonl')
            logger.info(f"load key text from {key_jsonl}")
            keys = [json.loads(item) for item in open(key_jsonl).readlines()]
            assert len(keys) == len(data), f"result file for key incomplete {key_jsonl}"
            for item1, item2 in zip(keys, data):
                assert item1["id"] == item2["id"], "Inconsistent data id"
        else:      # empty placeholder
            keys = [{}] * len(data)

        data_len = len(keys)

        os.makedirs(icl_config.output_dir, exist_ok=True)
        results_save_dir = os.path.join(icl_config.output_dir, dataset['dataset_name'])
        os.makedirs(results_save_dir, exist_ok=True)
        logger.info(f"Results will be saved into {results_save_dir}")

        # load results or newly
        jsonl_file = os.path.join(results_save_dir, 'retrieval.jsonl')
        medium_res = extract_matching_valid_lines(data, jsonl_file)
        writer = rewrite_jsonl_with_valid_lines(medium_res, jsonl_file)

        logger.info(f"Note that in current mode, examples are retrieved by text if mode is 'text', else will using response from key_text_dir argument. Current mode: in domain {icl_config.in_domain_example_mode}, cross domain {icl_config.cross_domain_example_mode}")

        def prefix_preprocess(key_item, data_item):
            response = key_item["prediction"].strip()
            key = f"For sentence:\n{data_item['text']}\nSome possible corrections are as follows:\n{response}"
            return key
        
        def no_preprocess(key_item, data_item):
            return key_item["prediction"].strip()
        
        def query_preprocess():
            if icl_config.key_preprocess == "prefix":
                return prefix_preprocess
            else:
                return no_preprocess

        # retrieve for every item in the dataset
        if len(medium_res) < data_len:
            for key_item, data_item in tqdm(zip(keys[len(medium_res):], data.select(range(len(medium_res), len(data)))), desc=f"{dataset['dataset_name']} retrieval"):
                if icl_config.in_domain_example_mode in ['text', 'bm25']:
                    key_in_domain = data_item["text"]
                else:
                    key_in_domain = query_preprocess()(key_item, data_item)
                if icl_config.cross_domain_example_mode in ['text', 'bm25']:
                    key_cross_domain = data_item["text"]
                else:
                    key_cross_domain = query_preprocess()(key_item, data_item)
                in_domain_examples = dataset_controller.specific_database[dataset['dataset_name']].select(query=key_in_domain)
                cross_domain_examples = dataset_controller.full_database.select(query=key_cross_domain)

                save_item = dict(data_item)
                save_item['key_in_domain'] = key_in_domain
                save_item['in_domain_examples'] = in_domain_examples
                save_item['key_cross_domain'] = key_cross_domain
                save_item['cross_domain_examples'] = cross_domain_examples

                writer.write(json.dumps(save_item, ensure_ascii=False) + '\n')
        
        writer.close()
