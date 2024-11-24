from settings import *
from copy import deepcopy

import os
import json
from typing import List, Dict
os.makedirs(PROCESSED_DATASETS_DIR, exist_ok=True)

def to_new_dataset_json(
        save_dir,
        trainset_items: List[Dict]=[],
        validset_items: List[Dict]=[],
        testset_items: List[Dict]=[],
):
    data_dir = os.path.join(PROCESSED_DATASETS_DIR, save_dir)
    os.makedirs(data_dir, exist_ok=True)

    print(f'Train/Valid/Test size: {len(trainset_items)}/{len(validset_items)}/{len(testset_items)}')    

    for splitset in [trainset_items, validset_items, testset_items]:
        if len(splitset) == 0:
            splitset.append(BLANK_ITEM)
    
    for splitset in [trainset_items, validset_items, testset_items]:
        for i in range(len(splitset)):
            if splitset == testset_items:
                if 'labels' in splitset[i]:
                    splitset[i]['label'] = splitset[i]['labels'][0]
            else:
                splitset[i]['label'] = splitset[i]['labels'][0]

    with open(os.path.join(data_dir, 'train.json'), 'w') as f:
        json.dump(trainset_items, f, ensure_ascii=False, indent=4)

    with open(os.path.join(data_dir, 'valid.json'), 'w') as f:
        json.dump(validset_items, f, ensure_ascii=False, indent=4)
    
    with open(os.path.join(data_dir, 'test.json'), 'w') as f:
        json.dump(testset_items, f, ensure_ascii=False, indent=4)


def generator_of_m2(
    m2_file
):
    skip_edits = {"noop", "UNK", "Um"}
    with open(m2_file, "r", encoding="utf-8") as f:
        idx_ex = 1
        src_sent, tgt_sent, corrections, offset = None, None, [], 0
        for idx_line, _line in enumerate(f):
            line = _line.strip()

            if len(line) > 0:
                prefix, remainder = line[0], line[2:]
                if prefix == "S":
                    # there are definitedly some ill-formed m2 file that do not have a \n between Sentences
                    if src_sent:
                        yield idx_ex, {
                            "id": idx_ex,
                            "src_tokens": src_sent,
                            "tgt_tokens": tgt_sent,
                            "text": ' '.join(src_sent),
                            "label": ' '.join(tgt_sent),          
                            "corrections": corrections
                        }
                        src_sent, tgt_sent, corrections, offset = None, None, [], 0
                        idx_ex += 1

                    src_sent = remainder.split(" ")
                    tgt_sent = deepcopy(src_sent)

                elif prefix == "A":
                    annotation_data = remainder.split("|||")
                    idx_start, idx_end = map(int, annotation_data[0].strip().split(" "))
                    edit_type, edit_text = annotation_data[1], annotation_data[2]
                    if edit_type in skip_edits:
                        continue

                    formatted_correction = {
                        "idx_src": list(range(idx_start, idx_end)),
                        "idx_tgt": [],
                        "corr_type": edit_type
                    }
                    annotator_id = int(annotation_data[-1])
                    assert annotator_id == 0, annotator_id

                    removal = len(edit_text) == 0 or edit_text == "-NONE-"
                    if removal:
                        for idx_to_remove in range(idx_start, idx_end):
                            del tgt_sent[offset + idx_to_remove]
                            offset -= 1

                    else:  # replacement/insertion
                        edit_tokens = edit_text.split(" ")
                        len_diff = len(edit_tokens) - (idx_end - idx_start)

                        formatted_correction["idx_tgt"] = list(
                            range(offset + idx_start, offset + idx_end + len_diff))
                        tgt_sent[offset + idx_start: offset + idx_end] = edit_tokens
                        offset += len_diff

                    corrections.append(formatted_correction)

            else:  # empty line, indicating end of example
                if not src_sent:
                    continue
                yield idx_ex, {
                    "id": idx_ex,
                    "src_tokens": src_sent,
                    "tgt_tokens": tgt_sent,
                    "text": ' '.join(src_sent),
                    "label": ' '.join(tgt_sent),          
                    "corrections": corrections
                }
                src_sent, tgt_sent, corrections, offset = None, None, [], 0
                idx_ex += 1
            
        if src_sent:
            yield idx_ex, {
                "id": idx_ex,
                "src_tokens": src_sent,
                "tgt_tokens": tgt_sent,
                "text": ' '.join(src_sent),
                "label": ' '.join(tgt_sent),          
                "corrections": corrections
            }


def read_data_from_m2(
    m2_file
):
    data_list = []
    for i, item in generator_of_m2(m2_file):
        data_list.append(
            {
                "id": item["id"],
                "text": item["text"],
                "labels": [item["label"]]
            }
        )
    return data_list
