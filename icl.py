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

from data.icl_dataset import ICLDataset
from data.group_dataset import GroupDataset, InstructionDataset
from llm.pipeline import TextGeneration
from data.check_jsonl import extract_matching_valid_lines, rewrite_jsonl_with_valid_lines
from utils.log import setup_log, log_config
from utils.random import init_seed



logger = logging.getLogger(__name__)


def gec_extract_answer(string, original_text, start, end):
    if start not in string or 'No errors found' in string:
        return original_text
    
    # Bloomz bug: the last character will not be included in the result
    if len(end) >= 5:
        end = end[:-1]

    if end not in string:
        return original_text
    answer = string.split(start)[1].split(end)[0]
    answer = answer.strip()

    if answer == '' or answer == ' ':
        return original_text
    return answer

def gec_split_answer(string:str, original_text:str, splitter:str):
    splits = string.split(splitter)
    if len(splits) >= 3:
        # normal situation
        answer = splits[1].strip()
        if 0.5 <= len(answer) / len(original_text) <= 2.0:
            return answer
        else:
            return original_text
    else:
        return original_text


def start_end_extraction(string, start, end):
    if start:
        if string.find(start) != -1:
            string = string.split(start, 1)[1]
    if end:
        if string.find(end) != -1:
            string = string.split(end, 1)[0]
    return string


def postprocess(response, data_item, answer_start, answer_end, icl_config):
    # form result
    result_item = {'id': data_item['id'], 'text': data_item['text']}
    if 'label' in data_item:
        result_item['label'] = data_item["label"]

    if icl_config.postprocess == 'gec':
        # gec/fewshot postprocess
        answer = answer_start + ' ' + response
        answer = gec_extract_answer(answer, data_item['text'], answer_start, answer_end)
        result_item['prediction'] = answer
    elif icl_config.postprocess == 'splitter':
        answer = gec_split_answer(response, data_item['text'], '```')
        result_item['prediction'] = answer
    elif icl_config.postprocess == 'normal':
        answer = start_end_extraction(response, answer_start, answer_end)
        result_item['prediction'] = answer
    elif icl_config.postprocess == 'no':
        pass
    else:
        raise NotImplementedError(f'Unknown postprocess mode: {icl_config.postprocess}')
    
    result_item['response'] = response
    result_item['sentence'] = item['sentence']

    return result_item


if __name__ == "__main__":
    parser = TrlParser((ModelConfig, DataConfig, ICLConfig))
    model_config, data_config, icl_config = parser.parse_args_and_config()

    init_seed(icl_config.seed)

    setup_log(icl_config.output_dir)

    ## Forced parameter modification
    if 'qalb' in data_config.datasets:
        logger.info("[WARNING] For QALB datasets, we will set error examples (current for cross_domain_example_num) to 0.")
        icl_config.cross_domain_example_num = 0

    logger.info(f"Model arguments:")
    log_config(model_config, logger)
    logger.info(f"Data arguments:")
    log_config(data_config, logger)
    logger.info(f"In-context learning arguments:")
    log_config(icl_config, logger)

    pipe = TextGeneration(
        model_name=model_config.model_name_or_path,
        max_new_tokens=icl_config.max_new_tokens,
        do_sample=icl_config.do_sample,
        temperature=icl_config.temperature,
        top_k=icl_config.top_k,
        top_p=icl_config.top_p,
        return_full_text=False,
        num_return_sequences=1,
        stop_string=None,
    )

    # get dataset of ICL instructions
    dataset_controller = ICLDataset(data_args=data_config, model_args=model_config, icl_args=icl_config)

    # infer for every dataset
    for dataset in dataset_controller.get_datasets():
        data = dataset['dataset']
        answer_end = dataset['answer_end']
        answer_start = dataset['answer_start']

        pipe.reset_stop_string(stop_string=answer_end)

        logger.info(f"ICL Infer on dataset {dataset['dataset_name']}, by prompt {dataset['prompt_name']}")
        os.makedirs(icl_config.output_dir, exist_ok=True)
        results_save_dir = os.path.join(icl_config.output_dir, dataset['dataset_name'])
        os.makedirs(results_save_dir, exist_ok=True)
        logger.info(f"Results will be saved into {results_save_dir}")

        sentences = [item['sentence'] for item in data]
        logger.info("Data Example:\n" + str(random.choice(sentences)))

        jsonl_file = os.path.join(results_save_dir, 'predictions.jsonl')

        medium_res = extract_matching_valid_lines(data, jsonl_file)

        # API-based generation
        if pipe.mode == "API" and len(medium_res) != len(data):
            # generate for all data, cache in api_cache.json
            api_res_cache_file = os.path.join(results_save_dir, 'api_cache.json')
            results = pipe(data, api_res_cache_file)
            # convert to standard results
            with open(jsonl_file, 'w') as f:
                for i, item in enumerate(data):
                    assert results[i]["id"] == item["id"]
                    response = results[i]["result"].strip()
                    result_item = postprocess(response, item, answer_start, answer_end, icl_config)
                    f.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                    f.flush()


        # judge the result state now to continue on current results.
        medium_res = extract_matching_valid_lines(data, jsonl_file)
        writer = rewrite_jsonl_with_valid_lines(medium_res, jsonl_file)

        # Predict and save
        if len(medium_res) < len(data):
            for item in tqdm(data.select(range(len(medium_res), len(data))), desc=dataset["dataset_name"]):
                output = pipe(item['sentence'])
                response = output[0]['generated_text']

                result_item = postprocess(response, item, answer_start, answer_end, icl_config)
                
                writer.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                writer.flush()
        
        writer.close()
