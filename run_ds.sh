NUM_GPUS=1

MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct

FP16_LOSS_SCALE_WINDOW=100
FP16_INITIAL_SCALE_POWER=7
FP16_HYSTERESIS=1

REDUCE_BUCKET_SIZE=1e5
GRADIENT_CLIPPING=0.9

PER_DEVICE_TRAIN_BATCH_SIZE=4
NUM_TRAIN_ITERATIONS=3
GRADIENT_ACCUMULATION_STEPS=1
MAX_SEQ_LENGTH=1024

LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
BETA_0=0.9
BETA_1=0.95
SUB_GROUP_SIZE=0

STEP_PER_PRINT=10
SEED=42

ENABLE_LIGER_ROPE=1
ENABLE_LIGET_SWIGLU=1
ENABLE_LIGER_RMS=1
ENABLE_FLCE=1

# ensure the cache is clean
rm -rf ~/.cache/torch_extensions/

deepspeed --num_gpus $NUM_GPUS deepspeed_liger_kernel.py --model_name $MODEL_NAME \
    --fp16_loss_scale_window $FP16_LOSS_SCALE_WINDOW --fp16_initial_scale_power $FP16_INITIAL_SCALE_POWER --fp16_hysteresis $FP16_HYSTERESIS \
    --reduce_bucket_size $REDUCE_BUCKET_SIZE --gradient_clipping $GRADIENT_CLIPPING \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --num_train_iterations $NUM_TRAIN_ITERATIONS --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS --max_seq_len $MAX_SEQ_LENGTH \
    --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY --beta_0 $BETA_0 --beta_1 $BETA_1 --sub_group_size $SUB_GROUP_SIZE \
    --step_per_print $STEP_PER_PRINT --seed $SEED \
    --enable_liger_rope $ENABLE_LIGER_ROPE --enable_liger_swiglu $ENABLE_LIGET_SWIGLU --enable_liger_rms $ENABLE_LIGER_RMS --enable_liger_flce $ENABLE_FLCE \
    --gradient_checkpointing \
    # --is_flash_attn