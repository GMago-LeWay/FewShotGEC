from settings import *
from base import to_new_dataset_json, read_data_from_m2
import os
import json
import xml.etree.ElementTree as ET

NAME = 'estgec'

dataset_dir = get_multilingual_raw_data_dir(NAME)

l2_corpus_path = os.path.join(dataset_dir, 'Tartu_L2_corpus/Tartu_L2_learner_corpus_parallel.txt')
test_m2 = os.path.join(dataset_dir, 'Tartu_L1_corpus/test/test_m2.txt')


tree = ET.parse(l2_corpus_path)
root = tree.getroot()


train_items = []

for index, mistake in enumerate(root.findall('mistake'), start=1):
    item = {
        'id': index,
        'text': mistake.find('original').text.strip(),
        'labels': []
    }
    
    for correction in mistake.findall('correction'):
        item['labels'].append(correction.text.strip())
    
    train_items.append(item)

# read l1 corpus test set
test_items = read_data_from_m2(test_m2)

to_new_dataset_json(
    NAME,
    trainset_items=train_items,
    validset_items=[],
    testset_items=test_items
)
