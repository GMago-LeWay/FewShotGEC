from settings import *
from base import to_new_dataset_json, read_data_from_m2
import os
import json

NAME = 'rulec'

dataset_dir = get_multilingual_raw_data_dir(NAME)

train_m2 = os.path.join(dataset_dir, 'RULEC-GEC.dev.M2')
dev_m2 = os.path.join(dataset_dir, 'RULEC-GEC.dev.M2')
test_m2 = os.path.join(dataset_dir, 'RULEC-GEC.test.M2')

train_items = read_data_from_m2(train_m2)
valid_items = read_data_from_m2(dev_m2)
test_items = read_data_from_m2(test_m2)

to_new_dataset_json(
    NAME,
    trainset_items=train_items,
    validset_items=valid_items,
    testset_items=test_items
)
