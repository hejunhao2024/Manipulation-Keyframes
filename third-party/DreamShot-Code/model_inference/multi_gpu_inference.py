import os
import torch
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from dataclasses import dataclass, field
import transformers

@dataclass
class TrainConfig:
    gpu_num: int = 8
    total_num: int = 8
    start_cnt: int = 0

    model_name: str = "Wan2.1-T2V-A14B"
    model_path: str = ""
    model_path2: str = "None"
    rl_model_path: str = "None"
    output_dir: str = ""
    sample_id: str = "None"
    shot_mode: str = "ref2shot"
    seed: int = 42
    resolution: int = 720

    img_cfg_scale: float = 2
    cfg_scale: float = 3

    run_py: str = "inference_dreamshot.py"

    debug: bool = False
    use_last_ref: bool = False
    use_phase_offset: str = "None"

    target_frame: int = 6
    context_frame: int = 0

    num_inference_steps: int = 30
    lightx2v_model_path: str = "None"

    vistory_json_path: str = "None"

    data_root: str = "None"

def main():
    parser = transformers.HfArgumentParser(TrainConfig)
    config: TrainConfig = parser.parse_args_into_dataclasses()[0]
    with tqdm(range(config.gpu_num)) as pbar:
        def map_func(gpu_id):
            sub_idx = gpu_id + config.start_cnt
            cmd = f'CUDA_VISIBLE_DEVICES={gpu_id % config.gpu_num} python3 model_inference/{config.run_py} --sub_idx {sub_idx} --total_num {config.total_num} \
                --model_name {config.model_name} \
                --sample_id {config.sample_id} \
                --model_path {config.model_path} \
                --model_path2 {config.model_path2} \
                --shot_mode {config.shot_mode} \
                --output_dir {config.output_dir} \
                --seed {config.seed} \
                --resolution {config.resolution} \
                --use_last_ref {config.use_last_ref} \
                --use_phase_offset {config.use_phase_offset} \
                --img_cfg_scale {config.img_cfg_scale} \
                --cfg_scale {config.cfg_scale} \
                --target_frame {config.target_frame} \
                --context_frame {config.context_frame} \
                --num_inference_steps {config.num_inference_steps} \
                --lightx2v_model_path {config.lightx2v_model_path} \
                --rl_model_path {config.rl_model_path} \
                --vistory_json_path {config.vistory_json_path} \
                --data_root {config.data_root}'
                
            os.system(command=cmd)
            pbar.update()

        pool = ThreadPool(config.gpu_num)
        pool.map(map_func, range(config.gpu_num))
        pool.close()

if __name__ == '__main__':
    main()