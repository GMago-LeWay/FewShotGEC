# CUDA_VISIBLE_DEVICES="2,3"

CUDA_VISIBLE_DEVICES="2,3" accelerate launch --config_file deepspeed_zero0.yaml --num_processes 2 \
sft.py \
    --model_name_or_path="/data/xxx/models/bart-large" \
    --datasets="nucle,fce,wilocness,hsk,falko_merlin,cowsl2h,estgec,rogec,rulec,gecturk" \
    --prompts="default" \
    --dataset_text_field="sentence" \
    --trust_remote_code=true \
    --report_to="wandb" \
    --output_dir="results/sft_example" \
    --learning_rate=2e-5 \
    --lr_scheduler_type="cosine" \
    --warmup_ratio=0.05 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=8 \
    --max_seq_length=512 \
    --logging_steps=10 \
    --save_steps=100 \
    --eval_strategy="steps" \
    --eval_steps=100 \
    --num_train_epochs=2 \
    --gradient_checkpointing

    # --max_steps=-1 \
    # --use_peft \
    # --lora_r=64 \
    # --lora_alpha=16 \

    # --weight_decay=0.01 \
    # --adam_beta1=0.9 \
    # --adam_beta2=0.95 \
    # --optim='adamw_torch_fused' \

    # --hub_token="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" \
    # --push_to_hub \
