#!/bin/bash

# Define the fixed parameters
DEVICE="1"
DIR="experiments/icl_gpt4_temp"

# Define the datasets and databases
DATASETS=("conll14" "bea19" "falko_merlin" "rulec" "estgec" "rogec" "qalb2014" "mucgec" "nlpcc18" "fcgec")
DATABASES=("wilocness" "wilocness" "falko_merlin" "rulec" "estgec" "rogec" "qalb2014" "hsk" "hsk" "hsk")

# Run the script for each combination
for ((i=0; i<${#DATASETS[@]}; i++)); do
    DATASET=${DATASETS[$i]}
    DATABASE=${DATABASES[$i]}

    # Execute the script with the given parameters
    bash scripts/config_generation.sh "$DEVICE" "$DATASET" "$DATABASE" "$DIR"
done
