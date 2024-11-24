import logging
import random
from datasets import Dataset, Features, Value, Split, concatenate_datasets
from .dataset_processor.settings import BLANK_ITEM

from .dataset import GeneralDataset, InstructionDataset

logger = logging.getLogger(__name__)

class GroupDataset:
    def __init__(self, data_args, model_args) -> None:
        self.args = data_args
        self.model_args = model_args

        # check datasets and prompts
        self.dataset_names = data_args.datasets.split(',')
        self.prompt_names = data_args.prompts.split(',')

        logger.info(f"Dataset chosen: {self.dataset_names}")
        logger.info(f"Prompt chosen: {self.prompt_names}")

        assert len(self.dataset_names) == len(self.prompt_names) or len(self.prompt_names) == 1, "Unmatched length of dataset and prompt. They should have an equal num or prompt num is 1."

        # construct instruction dataset
        if len(self.prompt_names) == 1:
            self.prompt_names *= len(self.dataset_names)
        self.instructions_datasets = [InstructionDataset(data_args, model_args, dataset_name, prompt_name) for dataset_name, prompt_name in zip(self.dataset_names, self.prompt_names)]
        

    def get_dataset(self, split, shuffle_seed=None):
        dataset_list = [item.get_dataset(split) for item in self.instructions_datasets]
        def add_from_key(example, dataset_name):
            example['from'] = dataset_name
            return example
        dataset_list = [ds.map(lambda x: add_from_key(x, self.dataset_names[i])) for i, ds in enumerate(dataset_list)]
        dataset_list = [ds for ds in dataset_list if not (ds[0]['text'] == BLANK_ITEM['text'] and len(ds)==1)]

        included_dataset_sources = [ds[0]['from'] for ds in dataset_list]
        included_dataset_length = [len(ds) for ds in dataset_list]

        dataset_info = [f"{dataset}: {num}" for dataset, num in zip(included_dataset_sources, included_dataset_length)]
        dataset_info = '\n'.join(dataset_info)
        logger.info(f"In the final {split} dataset, the components are: \n{dataset_info}")

        concatenated_dataset = concatenate_datasets(dataset_list)

        concatenated_dataset = concatenated_dataset.shuffle()

        logger.info(f"Dataset example: {concatenated_dataset[0]}")

        return concatenated_dataset
    

    def __getitem__(self, key):
        if key == Split.TRAIN:
            return self.get_dataset('train')
        elif key == Split.VALIDATION:
            return self.get_dataset('valid')
        elif key == Split.TEST:
            return self.get_dataset('test')
        else:
            raise AttributeError(f"'InstructionDataset' object has no attribute '{key}'")
        
