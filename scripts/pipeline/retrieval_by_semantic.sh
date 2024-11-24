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

RESULT_DIR=$DIR_RETRIEVE_BY_SEMANTIC

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python icl_retrieval.py \
    --datasets="$DATASET" \
    --prompts="" \
    --trust_remote_code=true \
    --output_dir="$RESULT_DIR" \
    --max_new_tokens=512 \
    --model_name_or_path="$MODEL" \
    --in_domain_example_num=$EXAMPLE_NUM_ERROR_RETRIEVAL \
    --cross_domain_example_num=$EXAMPLE_NUM_CORRECT_RETRIEVAL \
    --in_domain_example_mode="text" \
    --cross_domain_example_mode="text" \
    --in_domain_filter="error" \
    --cross_domain_filter="correct" \
    --database \
    --database_dir=".cache" \
    --medium_result_dir=".cache/llm" \
    --assist_model="" \
    --assist_prompt="" \
    --retrieval_mode="single_key" \
    --embedding_model="$EMBED_MODEL" \
    --embedding_batch_size=128 
