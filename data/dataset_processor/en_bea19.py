from settings import *
from base import to_new_dataset_json, read_data_from_m2
import os
import json

NAME = 'bea19'

dataset_dir = get_multilingual_raw_data_dir(NAME)

bea19_input_file = os.path.join(dataset_dir, 'test/ABCN.test.bea19.orig')

    
test_data_list = []

# CoNLL14 data, item id {i}_conll14
# BEA-19 test data(W&I Locness test data) item id {i}_BEA19
BEA19_texts = open(bea19_input_file, 'r', encoding="utf-8").readlines()
for i, line in enumerate(BEA19_texts):
    test_data_list.append(
        {
            "id": f'{i}_bea19',
            "text": line.strip(),
            "src_tokens": line.strip().split(" "),
        }
    )

bea19_num = len(test_data_list)

to_new_dataset_json(
    NAME,
    trainset_items=[],
    validset_items=[],
    testset_items=test_data_list,
)
