set -x
# Definite the parameters
# dataset settings
# check yaml settings
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path-to-yaml-file>"
    exit 1
fi

# get yaml path from arg
YAML_FILE=$1

# set environment varialbles by print export commands.
source <(python yaml_load.py $YAML_FILE)

# Check settings
cat $YAML_FILE

## Stage1: run database prepare
# bash scripts/pipeline/prepare_icl.sh $YAML_FILE

## Stage2: get baseline results
## run random ICL
bash scripts/pipeline/infer_icl_random.sh $YAML_FILE

# retrieve by semantic
bash scripts/pipeline/retrieval_by_semantic.sh $YAML_FILE

## normal semantic icl
bash scripts/pipeline/infer_icl_semantic.sh $YAML_FILE

# retrieve by bm25
bash scripts/pipeline/retrieval_by_bm25.sh $YAML_FILE

## normal bm25 icl
bash scripts/pipeline/infer_icl_bm25.sh $YAML_FILE
