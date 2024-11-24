import os
import sys
import argparse
from transformers import AutoModel, AutoTokenizer

from config import MODEL_ROOT_DIR

def main(model_name):
    save_directory = os.path.join(MODEL_ROOT_DIR, model_name.replace("/", "_"))

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    print(f"Model and tokenizer saved to {save_directory}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]

    main(model_name)
