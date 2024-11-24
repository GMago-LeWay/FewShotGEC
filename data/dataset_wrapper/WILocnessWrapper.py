import os
import json
import logging
from tqdm import tqdm
import datasets

logger = logging.getLogger(__name__)


class WILocnessWrapper:
    def __init__(self, args, directory) -> None:
        '''
        BEA 19 WI&Locness dataset.
        train, valid split available; test split is EQUAL to validation.
        '''
        self.args = args
        self.data_dir = directory
        self.test_data_file = os.path.join(self.data_dir, 'test.json')

    def _load_json_and_formatted(self, file_path):
        data = json.load(open(file_path))
        if type(data) == list:
            assert len(data) != 0
            new_data = {}
            if 'id' not in data[0]:
                new_data['id'] = list(range(0, len(data)))
            for key in data[0]:
                new_data[key] = [item[key] for item in data]
            return new_data
        else:
            raise NotImplementedError()

    def get_dataset(self, split=None)-> dict:
        assert split in ['train', 'valid', 'test']
        if split in ['train', 'valid']:
            if split == 'valid':
                split = 'validation'
            data = datasets.load_dataset(
                'data/dataset_wrapper/WILocnessBuilder.py', 
                data_dir=self.data_dir, 
                name='all', 
                split=split,
                trust_remote_code=True,
            )
        else:
            file = self.test_data_file
            assert os.path.exists(file), 'Valid or test file does not exist.'
            json_data = self._load_json_and_formatted(file)
            data = datasets.Dataset.from_dict(json_data)
        return data
