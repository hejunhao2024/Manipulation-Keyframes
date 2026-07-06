CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
  --standalone \
  --nproc_per_node=4 \
  keyframegen/train/agibot_non_ar_train.py \
  --config configs/agibot/non_ar_dual_context.json


cd /media/datasets/yumi/hjh/Manipulation-Keyframes
  conda activate /media/datasets/yumi/hjh/conda_envs/keyframe

  DIFFSYNTH_SKIP_DOWNLOAD=True CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --num_gpus 4 \
    keyframegen/train/agibot_non_ar_train.py \
    --config configs/agibot/train/exp1_local_only_lora_rank128.json \
    2>&1 | tee runs/exp1_local_only_lora_rank128.log

    DIFFSYNTH_SKIP_DOWNLOAD=True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 deepspeed --num_gpus 6 \
    keyframegen/train/agibot_non_ar_train.py \
    --config configs/agibot/train/exp1_local_only_lora_rank128.json \
    2>&1 | tee runs/exp1_local_only_lora_rank128.log