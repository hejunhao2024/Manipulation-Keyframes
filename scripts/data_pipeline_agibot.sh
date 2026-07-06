mkdir -p /cache/logs

python -u /home/ma-user/work/annotate_ego4d_full_pipeline_detailed.py \
  --video-dir /cache/data/ego4d_goalstep_full/v2/full_scale \
  --output-root /cache/data/ego4d_goalstep_full/v2/annotated_ego4d_v2_detailed \
  --train-annotation /cache/data/ego4d_goalstep_full/v2/annotations/goalstep_train.json \
  --val-annotation /cache/data/ego4d_goalstep_full/v2/annotations/goalstep_val.json \
  --model Qwen3-VL-32B-Instruct \
  --endpoints http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1 \
  --sample-concurrency 10 \
  --request-concurrency 32 \
  --request-timeout 600 \
  --retries 2 \
  --sample-interval 1 \
  --frame-max-side 512 \
  --jpeg-quality 90 \
  --min-candidate-frames 60 \
  --window-size 15 \
  --window-stride 15 \
  --min-window-frames 5 \
  --selection-image-side 512 \
  --selection-max-tokens 900 \
  --min-keyframes 41 \
  --global-representative-frames 16 \
  --global-image-side 384 \
  --global-max-tokens 650 \
  --local-image-side 512 \
  --local-max-tokens 1400 \
  --aggregate-every 20 \
  --resume \
  --write-debug-json \
  2>&1 | tee /cache/logs/ego4d_full_detailed_$(date +%Y%m%d_%H%M%S).log




mkdir -p /cache/logs

set -o pipefail

PYTHONFAULTHANDLER=1 \
python -u /home/ma-user/work/annotate_ego4d_full_pipeline_detailed.py \
  --video-dir /cache/data/ego4d_goalstep_full/v2/full_scale \
  --output-root /cache/data/ego4d_goalstep_full/v2/annotated_ego4d_v2_detailed \
  --train-annotation /cache/data/ego4d_goalstep_full/v2/annotations/goalstep_train.json \
  --val-annotation /cache/data/ego4d_goalstep_full/v2/annotations/goalstep_val.json \
  --model Qwen3-VL-32B-Instruct \
  --endpoints http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1 \
  --sample-concurrency 2 \
  --request-concurrency 8 \
  --request-timeout 600 \
  --retries 2 \
  --sample-interval 1 \
  --frame-max-side 512 \
  --jpeg-quality 90 \
  --min-candidate-frames 60 \
  --window-size 15 \
  --window-stride 15 \
  --min-window-frames 5 \
  --selection-image-side 384 \
  --selection-max-tokens 900 \
  --min-keyframes 41 \
  --global-representative-frames 16 \
  --global-image-side 384 \
  --global-max-tokens 650 \
  --local-image-side 512 \
  --local-max-tokens 1400 \
  --aggregate-every 20 \
  --resume \
  --write-debug-json \
  2>&1 | tee /cache/logs/ego4d_full_detailed_$(date +%Y%m%d_%H%M%S).log

rc=${PIPESTATUS[0]}
echo "Python exit code: $rc"