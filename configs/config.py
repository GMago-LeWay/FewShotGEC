import os.path as osp
import json
import os
import logging
logger = logging.getLogger(__name__)

ROOT_WORK_DIR = json.load(open("configs/directory.json"))['root_work_directory']

DATA_ROOT_DIR =  osp.join(ROOT_WORK_DIR, 'datasets')
MODEL_ROOT_DIR =  osp.join(ROOT_WORK_DIR, 'models')


DATA_DIR_NAME = {
    'wilocness': "WILocness",
    'fce': "FCE",
    'nucle': "NUCLE",
    'lang8': "Lang8",
    'clang8': "clang8",
    'hybrid': "EnglishHybrid",
    'cowsl2h': "multilingual/cowsl2h",
    'falko_merlin': "multilingual/falko_merlin",
    'rogec': "multilingual/rogec",
    'rulec': "multilingual/rulec",
    'qalb2014': "multilingual/qalb2014",
    'qalb2015': "multilingual/qalb2015",
    'estgec': "multilingual/estgec",
    'hsk': "multilingual/hsk",
    'gecturk': "multilingual/gecturk",
    'kor_union': "multilingual/kor_union",
    'conll14': "multilingual/conll14",
    'bea19': "multilingual/bea19",
    'nlpcc18': "multilingual/nlpcc18",
}


def get_data_dir(single_dataset_name):
    assert single_dataset_name in DATA_DIR_NAME, f"{single_dataset_name} is not in the map in the DATA_DIR_NAME of configs/config.py"
    return os.path.join(DATA_ROOT_DIR, DATA_DIR_NAME[single_dataset_name])


