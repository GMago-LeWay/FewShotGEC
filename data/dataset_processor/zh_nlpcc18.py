from settings import *
from base import to_new_dataset_json
import os
import json

NAME = 'nlpcc18'

dataset_dir = get_multilingual_raw_data_dir(NAME)

source_file = os.path.join(dataset_dir, 'test/source.txt')

    
test_data_list = []

# id nlpcc_i
for i, line in enumerate(open(source_file).readlines()):
    test_data_list.append({"id": f"nlpcc_{i}", "text": line.strip()})

to_new_dataset_json(
    NAME,
    trainset_items=[],
    validset_items=[],
    testset_items=test_data_list,
)
