from settings import *
from base import to_new_dataset_json

import os
import json

NAME = 'falko_merlin'

dataset_dir = get_multilingual_raw_data_dir(NAME)

train_source_data = open(os.path.join(dataset_dir, 'fm-train.src')).readlines()
train_target_data = open(os.path.join(dataset_dir, 'fm-train.trg')).readlines()
dev_source_data = open(os.path.join(dataset_dir, 'fm-dev.src')).readlines()
dev_target_data = open(os.path.join(dataset_dir, 'fm-dev.trg')).readlines()
test_source_data = open(os.path.join(dataset_dir, 'fm-test.src')).readlines()
test_target_data = open(os.path.join(dataset_dir, 'fm-test.trg')).readlines()

assert len(train_source_data) == len(train_target_data)
assert len(dev_source_data) == len(dev_target_data)
assert len(test_source_data) == len(test_target_data)


train_items = []
for i, (src, tgt) in enumerate(zip(train_source_data, train_target_data)):
    train_items.append({"id": i, "text": src.strip(), "labels": [tgt.strip()]})

dev_items = []
for i, (src, tgt) in enumerate(zip(dev_source_data, dev_target_data)):
    dev_items.append({"id": i, "text": src.strip(), "labels": [tgt.strip()]})

test_items = []
for i, (src, tgt) in enumerate(zip(test_source_data, test_target_data)):
    test_items.append({"id": i, "text": src.strip(), "labels": [tgt.strip()]})


to_new_dataset_json(NAME, train_items, dev_items, test_items)
