import json
import os
import json
from typing import List, Dict, Union

def extract_matching_valid_lines(data: List[Dict[str, Union[int, str]]], jsonl_file: str) -> List[Dict[str, Union[int, str]]]:
    """
    Extracts valid lines from a JSONL file until the first mismatch or error during parsing.
    Stops processing upon encountering a mismatch in 'id' or a parsing error.
    
    :param data: List of dictionaries containing 'id' fields used for matching.
    :param jsonl_file: Path to the JSON Lines file to be read.
    :return: List of dictionaries representing valid lines up to the first mismatch or error.
    """
    valid_lines = []
    if not os.path.exists(jsonl_file):
        print(f"File {jsonl_file} does not exist.")
        return valid_lines

    with open(jsonl_file, 'r') as f:
        for idx, line in enumerate(f):
            if idx >= len(data):  # Stop if we've reached the end of the data list.
                break
            
            try:
                json_item = json.loads(line.strip())
                if data[idx].get('id') == json_item.get('id'):  # Check 'id' match at the current index.
                    valid_lines.append(json_item)
                else:  # Break on first mismatch.
                    break
            except json.JSONDecodeError:
                # On error during parsing, also break the loop.
                print(f"JSON decoding error encountered at line {idx + 1}. Terminating extraction.")
                raise BrokenPipeError(f"The incomplete file in {jsonl_file}")
                break

    return valid_lines

# Usage remains the same:
# matched_lines_or_error = extract_matching_valid_lines_until_mismatch_or_error(data_list, '/path/to/results/predictions.jsonl')
# print(matched_lines_or_error)

def rewrite_jsonl_with_valid_lines(valid_lines: List[Dict[str, Union[int, str]]], jsonl_file: str) -> None:
    """
    Rewrites the JSONL file with the provided valid lines, and returns the file pointer for further appending.
    
    :param valid_lines: List of dictionaries representing the valid JSON items to be written.
    :param jsonl_file: Path to the JSON Lines file to be rewritten.
    """
    # Ensure the file is opened in a mode that allows both writing and appending ('w+')
    f = open(jsonl_file, 'w+')
    for item in valid_lines:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
    # After writing all valid lines, the file pointer is at the end of the file, ready for appending.
    return f
