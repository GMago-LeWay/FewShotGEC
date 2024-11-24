export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Check if the first argument is provided and not empty
if [ "$#" -eq 1 ] && [ -n "$1" ]; then
    # Get YAML file path from the first argument
    YAML_FILE="$1"

    # Set environment variables by sourcing the commands generated by the Python script
    source <(python yaml_load.py "$YAML_FILE")
else
    echo "<<Warning>>: YAML file path is not provided or is empty. Please make sure that you have passed the required enviromental variables."
fi

# conversion
RESULT_DIR=$RESULT_DIR_ICL_RANDOM
IFS=':' read -r first_part second_part <<< "$DATASET"
TEST_DATASET_NAME=${first_part}

# delete old results
# rm -r $RESULT_DIR

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python icl.py \
    --datasets="$DATASET" \
    --prompts="$PROMPT_ICL" \
    --trust_remote_code=true \
    --output_dir=$RESULT_DIR \
    --max_new_tokens=512 \
    --postprocess="gec" \
    --model_name_or_path="$MODEL" \
    --in_domain_example_num=$EXAMPLE_NUM_ERROR \
    --cross_domain_example_num=$EXAMPLE_NUM_CORRECT \
    --in_domain_filter="error" \
    --cross_domain_filter="correct" \
    --in_domain_example_mode="default" \
    --cross_domain_example_mode="default" \
    --seed="111"

    # --dialogue_form \
# add --bf16 if use bfloat16


# evaluate
python evaluate_gec.py \
    --dir="${RESULT_DIR}/${TEST_DATASET_NAME}" \
    --dataset="${TEST_DATASET_NAME}" \
    --device="0"


case "$TEST_DATASET_NAME" in
    "conll14")
        python evaluators/m2scorer/scripts/m2scorer.py ${RESULT_DIR}/conll14/conll14.txt ${DATASETS_DIR}/multilingual_raw/EN-conll14st-test-data/noalt/official-2014.combined.m2 | tee ${RESULT_DIR}/conll14/conll14.score
        ;;
    "wilocness")
        python evaluators/m2scorer/scripts/m2scorer.py ${RESULT_DIR}/wilocness/conll14.txt ${DATASETS_DIR}/multilingual_raw/EN-conll14st-test-data/noalt/official-2014.combined.m2 | tee ${RESULT_DIR}/wilocness/conll14.score
        ;;
    "falko_merlin")
        python evaluators/m2scorer/scripts/m2scorer.py ${RESULT_DIR}/falko_merlin/falko_merlin-output-retokenized.txt ${DATASETS_DIR}/multilingual_raw/DE-FALKO-MERLIN/fm-test.m2 | tee ${RESULT_DIR}/falko_merlin/falko_merlin.score
        ;;
    "rulec")
        python evaluators/m2scorer/scripts/m2scorer.py ${RESULT_DIR}/rulec/rulec-output-retokenized.txt ${DATASETS_DIR}/multilingual_raw/RU-RULEC/RULEC-GEC.test.M2 | tee ${RESULT_DIR}/rulec/rulec.score
        ;;
    "estgec")
        python evaluators/m2scorer/scripts/m2scorer.py ${RESULT_DIR}/estgec/estgec-output-retokenized.txt ${DATASETS_DIR}/multilingual_raw/ET-estgec/Tartu_L1_corpus/test/test_m2.txt | tee ${RESULT_DIR}/estgec/estgec.score
        ;;
    *)
        echo "Unknown dataset name: $TEST_DATASET_NAME"
        ;;
esac
