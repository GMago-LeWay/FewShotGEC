# Few Shot System for Multilingual GEC
This is the repo for the paper "Explanation based In-Context Demonstrations Retrieval for Multilingual Grammatical Error Correction".

# How to use
## Configuration
Installation requires pytorch environment. After pytorch is installed, please use

```
pip install -r requirements.txt
```
to install packages.

Install spacy language packages through:

```
bash scripts/install_spacy.sh
```

## Directory Setting

Please get permission and download the raw GEC data into `/data/xxx/datasets/multilingual_raw` directory.
Our work have a recommend directory setting:

```
/data/xxx (your working directory)
|--- FewShotGEC   (This repo)
|--- models      (HuggingFace Models)
|--- datasets
   |--- multilingual_raw (Get permission and download the raw data, and rename to settings as in data/dataset_processor/settings.py)
   |--- multilingual (Processed dataset)
```

You can also use your own directory settings but you need to adjust `configs/directory.json` and `data/dataset_processor/settings` and all `.yaml` experimental configurations.

## Prepare Data
To process datasets into standard ones, you should use scripts in our repo to convert them.
```
python data/dataset_processor/en_conll14.py
python data/dataset_processor/zh_hsk.py
python data/dataset_processor/zh_nlpcc18.py
python data/dataset_processor/de_falko_merlin.py
python data/dataset_processor/ru_rulec.py
python data/dataset_processor/et_estgec.py
```


## Few-Shot baselines
We offer the scripts for all procedures including database construction, retrieval, inital detection and few-shot process.
Please refer to `scripts/pipeline` folder.

For example, you can run random-selection few-shot prediction by 
```
bash scripts/pipeline/infer_icl_random.sh experiments/icl_llama31/conll14.yaml
```
You can also run the ICL database (with grammatical error explanations, GEE) construction by
```
bash scripts/pipeline/prepare_icl.sh experiments/icl_llama31/conll14.yaml
```
Then you can get the dataset with GEE.

**Note that all the experiment shell scripts should be executed with a `.yaml` configuration file.**

For running baseline methods and proposed methods, you can refer to:
- `scripts/pipeline_baseline.sh`
- `scripts/pipeline_main.sh`

For reproducing, we offer the tasks that are used in main results:
```
bash experiments/task_llama31.sh
bash experiments/task_llama_baseline_methods.sh
...
```

# Brief view for the repo
- We have ICL scripts in the root directory, including `icl.py`, `icl_prepare.py` and `icl_retrieval.py`.
- And the `data/` include the datasets and the database implementation.
- The main experiments configuration files are saved in `experiments`