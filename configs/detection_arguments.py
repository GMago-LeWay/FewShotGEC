from dataclasses import dataclass, field
from typing import List, Optional
from trl.core import flatten_dict


@dataclass
class DetectionConfig:
    """
    Arguments which define the model and tokenizer to load.
    """
    mid_hidden_size: int = field(
        default=256,
        metadata={
            "help": ("Mid layer hidden size for detection output.")
        }
    )

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)
    
    def __post_init__(self):
        pass
