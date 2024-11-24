from settings import *
from base import to_new_dataset_json, read_data_from_m2
import os
import json

NAME = 'conll14'

dataset_dir = get_multilingual_raw_data_dir(NAME)

test_m2 = os.path.join(dataset_dir, 'noalt/official-2014.combined.m2')

    
test_data_list = []

# CoNLL14 data, item id {i}_conll14
with open(test_m2, 'r', encoding="utf-8") as f:
    idx_ex = 0
    src_sent, src_text = None, None
    for idx_line, _line in enumerate(f):
        line = _line.strip()
        if len(line) > 0:
            prefix, remainder = line[0], line[2:]
            if prefix == "S":
                src_text = remainder
                src_sent = remainder.split(" ")
            else:
                pass
        else:  # empty line, indicating end of example
            assert src_text != None
            test_data_list.append({
                "id": f'{idx_ex}_conll14',
                "text": src_text,
                "src_tokens": src_sent,
            })
            src_sent, src_text = None, None
            idx_ex += 1

conll14_num = len(test_data_list)

to_new_dataset_json(
    NAME,
    trainset_items=[],
    validset_items=[],
    testset_items=test_data_list,
)
