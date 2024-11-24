from dataclasses import dataclass, field
from typing import List, Optional
from trl.core import flatten_dict


@dataclass
class DataConfig:
    """
    Arguments which define the model and tokenizer to load.
    """
    # data_dir: str = field(
    #     default=None,
    #     metadata={
    #         "help": (
    #             "Dataset directory, can be automatically inferred by the `DATA_ROOT_DIR` and `DATA_DIR_NAME` by the datasets"
    #         )
    #     },
    # )
    datasets: str = field(
        default=None, 
        metadata={
            "help": "the dataset name. split by ',' like 'wilocness,hsk'; When using ICL mode, input a map for infer datasets like 'wilocness:wilocness,nucle,fce;hsk:hsk,mucgec', datasets after : will be set as in-domain retrieval database. "
        }
    )
    icl_datasets: str = field(
        default=None, 
        metadata={
            "help": "all the icl datasets applied in the full database (cross-domain) for ICL. split by ',' like 'wilocness,hsk'. If left blank in ICL scripts, the all ICL datasets will be set as all values of datasets map."
        }
    )
    prompts: str = field(
        default=None,
        metadata={
            "help": (
                "prompt name saved in TEMPLATE in data/instructions/template.py. The num is either 1 or matched with dataset_name. (multiple prompt should be split by comma)"
            )
        }
    )
    streaming: bool = field(
        default=False,
        metadata={"help": ("Load large dataset by streaming, only used in C4 now.")}
    )
    pre_split_length_for_infer: bool = field(
        default=False,
        metadata={"help": ("Preprocess the test set text by splitting by markers. Can be used in MuCGEC dataset.")}
    )
    valid_percent: float = field(
        default=None,
        metadata={
            "help": ("The validation set proportion of dataset when doing splitting for the whole dataset.")
        }
    )
    test_percent: float = field(
        default=None,
        metadata={
            "help": ("The test set proportion of dataset when doing splitting for the whole dataset.")
        }
    )
    target_mode: str = field(
        default='all',
        metadata={
            "help": ("The optimization mode of target text. 'all' for loss on all instructions; 'target' for loss on target, marked by sep token.")
        }
    )
    infer_mode: str = field(
        default=None,
        metadata={
            "help": ("The inference mode. None for inference on `test` split of dataset; `eval` for inference on `valid` split of dataset.")
        }
    )
    detection_prefix: str = field(
        default='',
        metadata={
            "help": ("Prefix prompt for detection.")
        }
    )
    detection_labels_num: int = field(
        default=3,
        metadata={
            "help": ("Detection label set. 2:KEEP, ERROR; 3:KEEP, INSERT, ERROR; 4:KEEP, INSERT, ERROR, DELETE")
        }
    )
    data_cache_dir: str = field(
        default=".cache",
        metadata={
            "help": ("The cache dir for data.")
        }
    )

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)
    
    def __post_init__(self):
        pass
