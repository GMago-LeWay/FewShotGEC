import logging
import random
import os
import glob
import json
from datasets import Dataset, Features, Value, Split, concatenate_datasets
from transformers import AutoTokenizer
from .dataset_processor.settings import BLANK_ITEM

from .dataset import GeneralDataset, InstructionDataset
from .editor import Editor
from .instructions.template import SuperPrompt
from .database import TextDataBase

logger = logging.getLogger(__name__)


class ZeroShotDataset:
    def __init__(self, data_args, model_args, icl_args) -> None:
        self.args = data_args
        self.model_args = model_args
        self.icl_args = icl_args

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        except:
            self.tokenizer = None
        # check datasets and prompts
        # datasets example for ZeroShot: wilocness,mucgec
        self.dataset_names = data_args.datasets.split(',')
        self.prompt_names = data_args.prompts.split(',')

        assert self.dataset_names is not None and self.prompt_names is not None

        logger.info(f"Dataset chosen: {self.dataset_names}")
        logger.info(f"Prompt chosen: {self.prompt_names}")

        assert len(self.dataset_names) == len(self.prompt_names) or len(self.prompt_names) == 1, "Unmatched length of dataset and prompt. They should have an equal num or prompt num is 1."

        # ICL datasets internal selection map
        self.icl_datasets_names_list = {}

        # Test dataset mapping
        self.test_datasets_map = {}
        for dataset_name in self.dataset_names:
            if data_args.infer_mode == 'eval':
                self.test_datasets_map[dataset_name] = GeneralDataset(data_args, model_args, dataset_name).get_standard_dataset_map('valid')['valid']
            else:
                self.test_datasets_map[dataset_name] = GeneralDataset(data_args, model_args, dataset_name).get_standard_dataset_map('test')['test']
        

        # construct instruction template
        if len(self.prompt_names) == 1:
            self.prompt_names *= len(self.dataset_names)

        # Add additional information by load previous generation results.
        if icl_args.last_generation_dir:
            self._init_info()

    def _init_info(self):
        # Test dataset mapping
        for dataset_name in self.dataset_names:
            jsonl_files = glob.glob(os.path.join(self.icl_args.last_generation_dir, dataset_name, '*.jsonl'))
            assert jsonl_files, f"The last generation results do not exist for {dataset_name} in {self.icl_args.last_generation_dir}"
            logger.info(f'Reading last generation results of {dataset_name} from {jsonl_files[0]}')
            last_generations = [json.loads(item.strip()) for item in open(jsonl_files[0]).readlines()]
            # check id
            assert len(self.test_datasets_map[dataset_name]) == len(last_generations), f"Unmatched last generation results from {jsonl_files[0]}"
            for item1, item2 in zip(self.test_datasets_map[dataset_name], last_generations):
                assert item1["id"] == item2["id"], f"Unmatched last generation results from {jsonl_files[0]}"

            last_generation_as = self.icl_args.last_generation_as.split(',')
            last_generation_mapping = {}
            for map_item in last_generation_as:
                key1, key2 = map_item.split(':')
                last_generation_mapping[key1] = key2
            
            def add_last(example, idx):
                for key in last_generation_mapping:
                    example[last_generation_mapping[key]] = last_generations[idx][key]
                return example
            
            # add information 
            self.test_datasets_map[dataset_name] = self.test_datasets_map[dataset_name].map(add_last, with_indices=True)
    

    def get_raw_datasets(self):
        datasets = []
        for test_dataset_name, prompt_name in zip(self.dataset_names, self.prompt_names):
            test_dataset = self.test_datasets_map[test_dataset_name]
            datasets.append({"dataset_name": test_dataset_name, "prompt_name": prompt_name, "dataset": test_dataset})
        return datasets

    def get_datasets(self):
        # instruction generaton
        datasets = []
        for test_dataset_name, prompt_name in zip(self.dataset_names, self.prompt_names):
            test_dataset = self.test_datasets_map[test_dataset_name]
            ## map the test dataset into instruction dataset with ICL examples
            template = SuperPrompt(prompt_name)
            template.set_editor(dataset_name=test_dataset_name)
            def _instruction_generate(item):
                sys_prompt, instruction_sentence = template.format(data_item=item)
                instruction_item = self.get_instruction_item(sys_prompt, instruction_sentence)
                item['sentence'] = instruction_item
                return item
            # 使用map函数应用转换
            reserved_columns = ['id', 'text', 'label', 'sentence']
            remove_columns = [key for key in list(test_dataset.column_names) if key not in reserved_columns]
            test_instruction_dataset = test_dataset.map(_instruction_generate, remove_columns=remove_columns)
            datasets.append({"dataset_name": test_dataset_name, "prompt_name": prompt_name, "dataset": test_instruction_dataset, "answer_start": template.get_answer_start(), "answer_end": template.get_answer_end()})
        return datasets
    

    def get_instruction_item(self, sys_prompt, instruction_sentence):
        dialogue = False
        if self.tokenizer == None or (self.icl_args.dialogue_form and self.tokenizer.chat_template):
            dialogue = True
        
        if dialogue:
            instruction_item = [
                {'role': "system", 'content': sys_prompt},
                {'role': "user", 'content': instruction_sentence}
            ]
        else:
            instruction_item = (sys_prompt + '\n' + instruction_sentence).strip()
        
        return instruction_item
