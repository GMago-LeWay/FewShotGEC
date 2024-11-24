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
bash scripts/pipeline/prepare_icl.sh $YAML_FILE

## Stage3: ICL by direct query generation (by prospective detection)
## run prospective detection by original text
bash scripts/pipeline/infer_probing.sh $YAML_FILE

# retrive based on prospective detection
bash scripts/pipeline/retrieval_by_probing.sh $YAML_FILE

# prospective detection retrieval ICL
bash scripts/pipeline/infer_icl_probing.sh $YAML_FILE
