import logging
import random
import os
import json
from datasets import Dataset, Features, Value, Split, concatenate_datasets
from transformers import AutoTokenizer
from .dataset_processor.settings import BLANK_ITEM

from .dataset import GeneralDataset, InstructionDataset
from .instructions.template import ICLPromptTemplate
from .database import TextDataBase

logger = logging.getLogger(__name__)


MODE_NEED_DATABASE = {
    'text',
    'relation',
    'rule_relation'
}

class DataBaseForICL:
    def __init__(self, icl_datasets, icl_config, cross_domain=False) -> None:
        assert icl_datasets, 'No dataset available for ICL.'
        self.icl_datasets = concatenate_datasets(icl_datasets)
        self.icl_config = icl_config
        self.cross_domain = cross_domain
        # self.errorneous_datasets = [item for item in self.icl_datasets if item['text'].strip() != item['label'].strip()]
        self.topk = icl_config.cross_domain_example_num if cross_domain else icl_config.in_domain_example_num
        self.strategy = icl_config.cross_domain_example_mode if cross_domain else icl_config.in_domain_example_mode

        self.prepare_basic_database()

        if cross_domain:
            logger.info(f"[DATABASE] Database for Cross Domain has been prepared. Example Selection mode {self.strategy}")
        else:
            logger.info(f"[DATABASE] Database for In Domain has been prepared. Example Selection mode {self.strategy}")

    def random_select(self):
        return random.choices(self.base.dataset, k=self.topk)
    
    # def random_error_select(self):
    #     return random.choices(self.errorneous_datasets, k=self.topk)

    def select(self, query):
        if self.strategy in ['random', 'default']:
            return self.random_select()
        # elif self.strategy == 'random_erroneous':
        #     return self.random_error_select()
        elif self.strategy in ['text', 'relation', 'edit', 'rule_relation', 'bm25']:
            return self.get_topk_from_database(query=query)
        else:
            raise NotImplementedError()
    
    def prepare_basic_database(self):
        need_database = True
        if self.strategy in ["random", "default"] or self.topk == 0:
            need_database = False
        if self.cross_domain:
            self.base = TextDataBase(self.icl_config, self.icl_datasets, "cross_domain", need_database=need_database)
        else:
            self.base = TextDataBase(self.icl_config, self.icl_datasets, 'in_domain', need_database=need_database)

    def get_topk_from_database(self, query):
        return self.base.retrieve(query=query)
        




class ICLDataset:
    def __init__(self, data_args, model_args, icl_args) -> None:
        self.args = data_args
        self.model_args = model_args
        self.icl_args = icl_args

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        except:
            self.tokenizer = None
        # check datasets and prompts
        # datasets example for ICL: wilocness:wilocness,nucle,fce;hsk:hsk,mucgec
        self.dataset_names = data_args.datasets.split(';')
        self.prompt_names = data_args.prompts.split(',')

        assert self.dataset_names is not None and self.prompt_names is not None

        logger.info(f"Dataset chosen: {self.dataset_names}")
        logger.info(f"Prompt chosen: {self.prompt_names}")

        assert len(self.dataset_names) == len(self.prompt_names) or len(self.prompt_names) == 1, "Unmatched length of dataset and prompt. They should have an equal num or prompt num is 1."

        # ICL datasets internal selection map
        self.icl_datasets_names_list = {}

        used_test_datasets = []
        used_train_datasets = []

        # add dataset relations
        for dataset_mapping in self.dataset_names:
            assert len(dataset_mapping.split(':')) == 2, f"Dataset map relation is not correct. {dataset_mapping}"
            dataset_name, icl_datasets = dataset_mapping.split(':')
            icl_datasets = icl_datasets.split(',')

            assert dataset_name not in self.icl_datasets_names_list, f"{dataset_name} duplicated in the datasets input."
            self.icl_datasets_names_list[dataset_name] = icl_datasets

            used_test_datasets.append(dataset_name)
            used_train_datasets.extend(icl_datasets)

        # train dataset dedup

        icl_train_datasets = list(set(used_train_datasets))
        # load all ICL datasets as full database if any
        if data_args.icl_datasets:
            train_datasets_in_args = data_args.icl_datasets.split(',')
            if set(train_datasets_in_args) == set(used_train_datasets):
                logger.info("No contradiction in the arguments involved in ICL.")
                icl_train_datasets = list(set(used_train_datasets))
            elif set(train_datasets_in_args).issubset(set(used_train_datasets)):
                logger.info(f"Warning: Arguments `datasets` cover more train dataset, global ICL selection will be executed in values of datasets map in arguments.")
                icl_train_datasets = list(set(used_train_datasets))
            elif set(used_train_datasets).issubset(set(train_datasets_in_args)):
                logger.info(f"Warning: Arguments `icl_datasets` cover more train dataset, global ICL selection will be executed in icl_datasets in arguments.")
                icl_train_datasets = list(set(train_datasets_in_args))
            else:
                raise ValueError('Contradiction in the arguments `datasets` and `icl_datasets`.')
            
        # test dataset dedup
        assert len(used_test_datasets) == len(list(set(used_test_datasets))), 'The ICL datasets to infer contain duplicated item.'

        # General dataset mapping    
        self.train_datasets_map = {}
        for dataset_name in icl_train_datasets:
            self.train_datasets_map[dataset_name] = GeneralDataset(data_args, model_args, dataset_name).get_standard_dataset_map('train')['train']
        
        # Test dataset mapping
        self.dataset_names = used_test_datasets
        self.test_datasets_map = {}
        for dataset_name in used_test_datasets:
            if data_args.infer_mode == 'eval':
                self.test_datasets_map[dataset_name] = GeneralDataset(data_args, model_args, dataset_name).get_standard_dataset_map('valid')['valid']
            else:
                self.test_datasets_map[dataset_name] = GeneralDataset(data_args, model_args, dataset_name).get_standard_dataset_map('test')['test']
        

        # construct instruction template
        if len(self.prompt_names) == 1:
            self.prompt_names *= len(self.dataset_names)

        if icl_args.examples_dir and os.path.exists(icl_args.examples_dir):
            self._init_examples()
        else:
            self._init_database()
    
    def _init_examples(self):
        # specific database for loaded results from example directory `examples_dir`
        self.specific_database = {}

        self.full_database = None

        for dataset_name in self.icl_datasets_names_list:
            examples_file = os.path.join(self.icl_args.examples_dir, dataset_name, 'retrieval.jsonl')
            self.specific_database[dataset_name] = [json.loads(item) for item in open(examples_file).readlines()]



    def _init_database(self):
        self.specific_database = {}
        
        dataset_list = list(self.train_datasets_map.values())
        dataset_list = [ds for ds in dataset_list if ds and not (ds[0]['text'] == BLANK_ITEM['text'] and len(ds)==1)]
      
        # init in-domain database
        for dataset_name in self.icl_datasets_names_list:
            self.specific_database[dataset_name] = DataBaseForICL(
                [self.train_datasets_map[key] for key in self.icl_datasets_names_list[dataset_name]], 
                self.icl_args,
                cross_domain=False
            )

        # init cross-domain database
        self.full_database = DataBaseForICL(dataset_list, self.icl_args, cross_domain=True)

    def normalize_similarities(self, similarity_list):
        if not similarity_list:
            return []

        # 提取相似度值
        similarities = [item['similarity'] for item in similarity_list]
        
        # 获取最小和最大值
        min_val = min(similarities)
        max_val = max(similarities)
        
        # 检查是否所有值都相同
        if min_val == max_val:
            # 如果所有值都相同，直接将它们设为 1
            for item in similarity_list:
                item['similarity'] = 1.0
        else:
            # 归一化处理
            for item in similarity_list:
                normalized_similarity = (item['similarity'] - min_val) / (max_val - min_val)
                item['similarity'] = normalized_similarity

    def get_dedup_examples(self, examples, num_limit):
        if not examples:
            return examples
        
        # check keys in example and normalize similarity
        keys = examples[0].keys()

        if 'similarity' in keys:
            self.normalize_similarities(examples)

        if 'id' in keys and 'data_item_id' in keys and 'similarity' in keys:
            # reorder by dedup
            data_item_map = {}
            for example in examples:
                if example["data_item_id"] not in data_item_map:
                    data_item_map[example["data_item_id"]] = example
                    data_item_map[example["data_item_id"]]["times"] = 1
                else:
                    data_item_map[example["data_item_id"]]["similarity"] += example["similarity"]
                    data_item_map[example["data_item_id"]]["times"] += 1

            dedup_examples = list(data_item_map.values())
            dedup_examples = sorted(dedup_examples, key=lambda x:x["similarity"], reverse=True)
            if len(dedup_examples) >= num_limit:
                return dedup_examples
            else:
                logger.info(f"[WARNING IN ICL] After dedup, only {len(dedup_examples)} demonstration left, which do not meet the requirement {num_limit}.")
                # add duplicate examples until num_limit, cover more data_item_id
                while len(dedup_examples) < num_limit:
                    for data_item_id in data_item_map:
                        if data_item_map[data_item_id]["times"] != 1:
                            dedup_examples.append(data_item_map[data_item_id])
                            data_item_map[data_item_id]["times"] -= 1
                return dedup_examples
            
        return examples

    def get_raw_datasets(self):
        datasets = []
        for test_dataset_name, prompt_name in zip(self.dataset_names, self.prompt_names):
            test_dataset = self.test_datasets_map[test_dataset_name]
            datasets.append({"dataset_name": test_dataset_name, "prompt_name": prompt_name, "dataset": test_dataset})
        return datasets
    
    def get_datasets_using_retrieval_results(self):
        datasets = []
        for test_dataset_name, prompt_name in zip(self.dataset_names, self.prompt_names):
            test_dataset = self.test_datasets_map[test_dataset_name]
            
            ## map the test dataset into instruction dataset with ICL examples
            template = ICLPromptTemplate(prompt_name)
            in_domain_topk = self.icl_args.in_domain_example_num
            cross_domain_topk = self.icl_args.cross_domain_example_num
            logger.info(f"Select {in_domain_topk} in-domain examples in ICL; {cross_domain_topk} cross-domain examples in ICL.")
            
            # check dataset consistency between test dataset and examples selection results
            examples_data = self.specific_database[test_dataset_name]
            assert len(test_dataset) == len(examples_data)
            for item1, item2 in zip(test_dataset, examples_data):
                assert item1['id'] == item2['id']

            # get examples
            test_instruction_items = []
            if_info_shown = False
            for item in examples_data:
                in_domain_examples = self.get_dedup_examples(item["in_domain_examples"], in_domain_topk)
                in_domain_examples = in_domain_examples[:in_domain_topk]
                cross_domain_examples = item["cross_domain_examples"][:cross_domain_topk]
                item['description'] = item["key_in_domain"]
                for item_in in in_domain_examples:
                    item_in['description'] = item_in["key"]
                for item_cross in cross_domain_examples:
                    item_cross['description'] = "No grammatical or spelling error is found in this sentence."
                if not if_info_shown:
                    logger.info("[INFO] Description of the predicted text is key_in_domain; Descriptions of the retrieved in-domain items is key.")
                    if_info_shown = True
                instruction_item = self.get_instruction(examples=in_domain_examples+cross_domain_examples, test_item=item, template=template)
                new_item = {"id": item["id"], "text": item["text"]}
                if "label" in item:
                    new_item['label'] = item['label']
                new_item['sentence'] = instruction_item
                test_instruction_items.append(new_item)
        
            keys = ['id', 'text', 'label', 'sentence'] if 'label' in test_instruction_items[0] else ['id', 'text', 'sentence']
            
            test_instruction_dataset = Dataset.from_dict({key: [example.get(key) for example in test_instruction_items] for key in keys})
            datasets.append({"dataset_name": test_dataset_name, "prompt_name": prompt_name, "dataset": test_instruction_dataset, "answer_start": template.get_answer_start(), "answer_end": template.get_answer_end()})
        return datasets


    def get_datasets(self):
        if self.icl_args.examples_dir:
            logger.info(f'[INFO] Input example data from {self.icl_args.examples_dir}')
            logger.info(f'[INFO] Due to the loaded examples, the settings of the example mode become invalid.')
            return self.get_datasets_using_retrieval_results()
        
        # normal retrieval
        logger.info('[INFO] No example data. Retrieve using original text when necessary.')
        datasets = []
        for test_dataset_name, prompt_name in zip(self.dataset_names, self.prompt_names):
            test_dataset = self.test_datasets_map[test_dataset_name]
            ## map the test dataset into instruction dataset with ICL examples
            template = ICLPromptTemplate(prompt_name)
            in_domain_topk = self.icl_args.in_domain_example_num
            cross_domain_topk = self.icl_args.cross_domain_example_num
            logger.info(f"Select {in_domain_topk} in-domain examples in ICL; {cross_domain_topk} cross-domain examples in ICL.")
            # get examples
            def _instruction_generate(item):
                in_domain_examples = self.specific_database[test_dataset_name].select(query=item["text"])
                cross_domain_examples = self.full_database.select(query=item["text"])
                instruction_item = self.get_instruction(examples=in_domain_examples+cross_domain_examples, test_item=item, template=template)
                item['sentence'] = instruction_item
                return item
            # 使用map函数应用转换
            reserved_columns = ['id', 'text', 'label', 'sentence']
            remove_columns = [key for key in list(test_dataset.column_names) if key not in reserved_columns]
            test_instruction_dataset = test_dataset.map(_instruction_generate, remove_columns=remove_columns)
            datasets.append({"dataset_name": test_dataset_name, "prompt_name": prompt_name, "dataset": test_instruction_dataset, "answer_start": template.get_answer_start(), "answer_end": template.get_answer_end()})
        return datasets
    

    def get_instruction(self, examples, test_item, template):
        converted_example_list = []
        for example_item in examples:
            value_dict = {'source': example_item['text'], 'target': example_item['label']}
            if 'description' in example_item:
                value_dict['description'] = example_item['description'].strip()
            converted_example_list.append(value_dict)
        
        if 'description' not in test_item:
            sys_prompt, instruction_sentence = template.format(examples_list=converted_example_list, source=test_item['text'])
        else:
            sys_prompt, instruction_sentence = template.format(examples_list=converted_example_list, source=test_item['text'], description=test_item['description'])

        dialogue = False
        if self.tokenizer == None or (self.icl_args.dialogue_form and self.tokenizer.chat_template):
            dialogue = True
        
        if dialogue:
            if self.tokenizer and 'System role not supported' in self.tokenizer.chat_template:
                instruction_item = [
                    {'role': "user", 'content': (sys_prompt + '\n' + instruction_sentence).strip()}
                ]
            else:
                instruction_item = [
                    {'role': "system", 'content': sys_prompt},
                    {'role': "user", 'content': instruction_sentence}
                ]
        else:
            instruction_item = (sys_prompt + '\n' + instruction_sentence).strip()
        
        return instruction_item
