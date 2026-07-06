cd /media/datasets/yumi/hjh/Manipulation-Keyframes

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
  --standalone \
  --nproc_per_node=4 \
  keyframegen/infer/agibot_non_ar_infer.py \
  --config configs/agibot/infer_non_ar_dual_context.json \
  2>&1 | tee /media/datasets/yumi/hjh/runs/agibot_non_ar_dual_rank128/inference/infer_$(date +%Y%m%d_%H%M%S).log