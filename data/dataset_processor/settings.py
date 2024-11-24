import os
import json

ROOT_WORK_DIR = json.load(open("configs/directory.json"))['root_work_directory']

# raw datasets directory
RAW_DATASETS_DIR = os.path.join(ROOT_WORK_DIR, 'datasets/multilingual_raw')

# process results directory
PROCESSED_DATASETS_DIR = os.path.join(ROOT_WORK_DIR, 'datasets/multilingual')

# blank item
BLANK_ITEM = {"id": "-1", "text": "Intentionally blank.", "labels": ["Intentionally blank."]}

# dataset dir
DATASET_DIR = {
    'qalb':  'AR-QALB-0.9.1',
    'falko_merlin': 'DE-FALKO-MERLIN',
    'cowsl2h': 'ES-cowsl2h',
    'estgec': 'ET-estgec',
    'rogec': 'RO-RoGEC',
    'rulec': 'RU-RULEC',
    'gecturk': 'TR-datasets',
    'hsk': 'ZH-nacgec_training',
    'kor_union': 'KR-Shared Dataset',
    'conll14': 'EN-conll14st-test-data',
    'bea19': 'EN-wi+locness',
    'nlpcc18': 'ZH-nlpcc18'
}


def get_multilingual_raw_data_dir(dataset_name):
    assert dataset_name in DATASET_DIR, f"{dataset_name} is not in the supported raw multilingual datasets."
    return os.path.join(RAW_DATASETS_DIR, DATASET_DIR[dataset_name])
