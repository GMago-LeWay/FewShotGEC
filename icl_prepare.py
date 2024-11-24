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
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset, Features, Value, Split

from data.icl_dataset import ICLDataset

from data.group_dataset import GroupDataset, InstructionDataset
from utils.log import setup_log, log_config
from utils.random import init_seed


logger = logging.getLogger(__name__)


def gec_extract_answer(string, original_text, start, end):
    if start not in string or 'No errors found' in string:
        return original_text
    if end not in string:
        return original_text
    answer = string.split(start)[1].split(end)[0]
    answer = answer.strip()

    if answer == '' or answer == ' ':
        return original_text
    return answer


if __name__ == "__main__":
    parser = TrlParser((ModelConfig, DataConfig, ICLConfig))
    model_config, data_config, icl_config = parser.parse_args_and_config()

    init_seed(icl_config.seed)

    if icl_config.medium_result_dir != icl_config.output_dir:
        print(f"Find medium result dir different from output dir in script of ICL preparation. Set output_dir equal to medium result dir {icl_config.medium_result_dir}")
        icl_config.output_dir = icl_config.medium_result_dir

    setup_log(icl_config.output_dir)

    logger.info(f"Model arguments:")
    log_config(model_config, logger)
    logger.info(f"Data arguments:")
    log_config(data_config, logger)
    logger.info(f"In-context learning arguments:")
    log_config(icl_config, logger)

    # get dataset of ICL instructions
    dataset_controller = ICLDataset(data_args=data_config, model_args=model_config, icl_args=icl_config)

    # infer for every dataset
    # for dataset in dataset_controller.get_datasets():
    #     pass
