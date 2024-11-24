from settings import *
from base import to_new_dataset_json, read_data_from_m2
import os
import json

NAME = 'hsk'

dataset_dir = get_multilingual_raw_data_dir(NAME)

src_file = os.path.join(dataset_dir, 'HSK/hsk.src')
tgt_file = os.path.join(dataset_dir, 'HSK/hsk.trg')

src_texts = open(src_file).readlines()
tgt_texts = open(tgt_file).readlines()

assert len(src_texts) == len(tgt_texts)


train_items = []

for i, (src, tgt) in enumerate(zip(src_texts, tgt_texts)):
    train_items.append(
        {
            "id": i,
            "text": src.strip(),
            "labels": [tgt.strip()]
        }
    )


to_new_dataset_json(
    NAME,
    trainset_items=train_items,
)
