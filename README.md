# Manipulation-Keyframes
模型架构：
1. local prompt only pipeline
diffsynth/models/wan_video_dit.py
diffsynth/pipelines/keyframe_local_context.py

2. dual prompt pipeline
diffsynth/models/wan_video_dit_dual_context.py

3. AR dual prompt pipeline
diffsynth/models/wan_video_dit_dual_context.py
diffsynth/pipelines/keyframe_dual_context.py

数据预处理： 
agibot or ego4d
keyframegen/data/encode_vae.py
keyframegen/data/encode_local_context.py
keyframegen/data/encode_dual_context.py


训练范式：
1. Agibot non-AR training
keyframegen/train/agibot_non_ar_train.py

2. Ego4d naive AR training
keyframegen/train/ego4d_naive_ar_train.py

3. Ego4d SVI AR training
keyframegen/train/edo4d_svi_train.py

推理：
1. Agibot non-AR inference
支持加载local/global or only local prompt
keyframegen/infer/infer_agibot_non_ar.py

2. Ego4d AR inference
keyframegen/infer/infer_ego4d_ar.py

配置文件
1. Agibot local-only configs
configs/

2. Agibot local-global configs

3. Ego4d naive AR configs

4. Ego4d SVI AR configs

评测（针对ckpt）
1. 