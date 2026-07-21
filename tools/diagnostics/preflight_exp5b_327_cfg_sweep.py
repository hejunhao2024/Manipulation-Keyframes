#!/usr/bin/env python3
import argparse, csv, json, math, os, sys, time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsynth.pipelines.keyframe_local_context import (
    WanVideoUnit_PromptEmbedder,
    WanVideoUnit_ImageEmbedderCLIP,
    WanVideoUnit_ImageEmbedderVAE,
)
from diffsynth.utils.data import save_video
from keyframegen.infer import infer_exp
from keyframegen.train import agibot_non_ar_train as trainer


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_json(p):
    with open(p, 'r', encoding='utf-8') as f: return json.load(f)


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    ensure_dir(path.parent)
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def tensor_compare(name, fresh, cached):
    fresh=fresh.detach().float().cpu(); cached=cached.detach().float().cpu()
    if tuple(fresh.shape) != tuple(cached.shape):
        return {'name':name,'fresh_shape':tuple(fresh.shape),'cached_shape':tuple(cached.shape),'shape_match':False}
    diff=(fresh-cached).abs()
    cos=[]
    if fresh.ndim >= 2 and fresh.shape[0] == 1 and fresh.shape[1] in (16,):
        for i in range(fresh.shape[1]):
            cos.append(float(F.cosine_similarity(fresh[:,i].reshape(1,-1), cached[:,i].reshape(1,-1)).item()))
    return {
        'name':name,
        'fresh_shape':tuple(fresh.shape),
        'cached_shape':tuple(cached.shape),
        'shape_match':True,
        'max_abs_diff':float(diff.max().item()),
        'mean_abs_diff':float(diff.mean().item()),
        'cosine_similarity':float(F.cosine_similarity(fresh.reshape(1,-1), cached.reshape(1,-1)).item()),
        'min_slot_cosine':min(cos) if cos else None,
        'mean_slot_cosine':sum(cos)/len(cos) if cos else None,
        'slot_cosines':cos,
    }


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__(); self.base=base; self.rank=int(rank); self.alpha=int(alpha); self.scale=self.alpha/self.rank
        self.dropout=nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        for p in self.base.parameters(): p.requires_grad_(False)
        self.lora_A=nn.Parameter(torch.empty(self.rank, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features, self.rank, dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    def forward(self,x):
        base=self.base(x); z=self.dropout(x).to(self.lora_A); z=z@self.lora_A.t(); z=z@self.lora_B.t(); return base + z.to(base)*self.scale


def inject_lora(model, adapter_cfg):
    replaced=[]
    for name,module in list(model.named_modules()):
        if not isinstance(module, nn.Linear): continue
        if not any(k in name for k in adapter_cfg['target_keywords']): continue
        if any(k in name for k in adapter_cfg.get('skip_keywords', [])): continue
        parent, child = infer_exp.get_parent_module(model, name)
        setattr(parent, child, LoRALinear(module, adapter_cfg['rank'], adapter_cfg['alpha'], adapter_cfg.get('dropout',0.0)))
        replaced.append(name)
    return replaced


def strict_load(pipe, cfg, checkpoint_path):
    replaced=inject_lora(pipe.dit, cfg['adapter'])
    ckpt=torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state=ckpt['trainable_state_dict']
    current=dict(pipe.dit.named_parameters())
    missing=[]; shape=[]; compatible={}
    groups={g:0 for g in ['self_attn_lora','cross_attn_shared_lora','cross_attn_text_lora','cross_attn_image_lora','norm3_full','img_emb_full']}
    for k,v in state.items():
        if k not in current: missing.append(k); continue
        if tuple(v.shape)!=tuple(current[k].shape): shape.append(k); continue
        compatible[k]=v.to(dtype=current[k].dtype)
        try:
            groups[trainer.classify_trainable_parameter(k)] += 1
        except Exception:
            pass
    unexpected=[k for k in current if False]
    if missing or shape or len(compatible)!=len(state) or any(v==0 for v in groups.values()):
        raise RuntimeError(f'strict load failed missing={missing[:5]} shape={shape[:5]} loaded={len(compatible)} state={len(state)} groups={groups}')
    msg=pipe.dit.load_state_dict(compatible, strict=False)
    return {'global_step':ckpt.get('global_step'),'checkpoint_tensors':len(state),'loaded':len(compatible),'skipped':0,'missing':missing,'shape_mismatch':shape,'groups':groups,'injected_lora':len(replaced),'load_missing_after_strict_false':len(msg.missing_keys)}


def save_contact(frames, path, thumb_width=192):
    thumbs=[]
    for im in frames:
        im=im.convert('RGB'); h=round(im.height*thumb_width/im.width); thumbs.append(im.resize((thumb_width,h)))
    cols=min(8,len(thumbs)); rows=(len(thumbs)+cols-1)//cols; label_h=22
    canvas=Image.new('RGB',(cols*thumb_width, rows*(thumbs[0].height+label_h)), 'white')
    from PIL import ImageDraw
    d=ImageDraw.Draw(canvas)
    for i,im in enumerate(thumbs):
        x=(i%cols)*thumb_width; y=(i//cols)*(im.height+label_h)
        d.text((x+4,y+4),f'{i:02d}',fill='black'); canvas.paste(im,(x,y+label_h))
    canvas.save(path)


def save_gt_pred_compare(sample, frames, out_dir, cfg_scale):
    sd=out_dir/f'cfg_{cfg_scale:g}'; ensure_dir(sd); pred_dir=sd/'pred'; ensure_dir(pred_dir)
    for i,fr in enumerate(frames): fr.save(pred_dir/f'{i:02d}.png')
    save_contact(frames, sd/'pred_contact.jpg')
    save_video(frames, str(sd/'pred_1fps.mp4'), fps=1, quality=5)
    gt=[]
    for p in sample['target_keyframes']:
        gt.append(Image.open(p).convert('RGB'))
    if len(gt)==len(frames):
        comp=[]
        for g,p in zip(gt,frames):
            gs=g.resize((512,512)); ps=p.resize((512,512)); c=Image.new('RGB',(1024,512),'black'); c.paste(gs,(0,0)); c.paste(ps,(512,0)); comp.append(c)
        save_contact(comp, sd/'compare_contact.jpg', thumb_width=256)
        save_video(comp, str(sd/'compare_1fps.mp4'), fps=1, quality=5)
    (sd/'meta.json').write_text(json.dumps({'cfg_scale':cfg_scale,'sample_id':sample['id'],'sample_dir':sample['sample_dir']},indent=2),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--sample-dir', default='/media/datasets/yumi/hjh/datasets/aigbot_final/327/648642-685032/648649')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args()
    out=Path(args.output_dir); ensure_dir(out); ensure_dir(out/'tables')
    cfg=load_json(args.config)
    cfg['checkpoint']['path']=args.checkpoint
    cfg['data']['expected_num_slots']=16
    cfg['inference']['output_dir']=str(out/'cfg_sweep')
    cfg['inference']['num_inference_steps']=50
    cfg['inference']['seed']=args.seed
    cfg['inference']['seed_per_sample']=False
    cfg['inference']['overwrite']=True

    sample=infer_exp.load_sample(args.sample_dir, 16)
    ann=load_json(Path(args.sample_dir)/'annotation.json')
    prompt_rows=[]
    for i,frame in enumerate(ann['frames']):
        chosen='generated_prompt' if frame.get('generated_prompt') else 'frame_prompt_en_compiled' if frame.get('frame_prompt_en_compiled') else 'frame_prompt_template_en'
        prompt_rows.append({'slot':i,'chosen_field':chosen,'generated_prompt':frame.get('generated_prompt',''),'frame_prompt_en_compiled':frame.get('frame_prompt_en_compiled',''),'frame_prompt_template_en':frame.get('frame_prompt_template_en',''),'actual_infer_prompt':sample['frame_prompts'][i]})
    write_csv(out/'tables'/'prompt_source.csv', prompt_rows)

    print('[stage] build pipe/load checkpoint', flush=True)
    pipe=infer_exp.build_pipe(cfg, 'local_only', local_rank=0)
    info=strict_load(pipe, cfg, args.checkpoint)
    (out/'checkpoint_strict_summary.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print('[checkpoint]', info, flush=True)

    print('[stage] fresh/cache compare', flush=True)
    inf=cfg['inference']; input_image=Image.open(sample['image']).convert('RGB')
    num_frames=1+(16-1)*4
    with torch.no_grad():
        punit=WanVideoUnit_PromptEmbedder(); fresh_context=punit.process(pipe=pipe,prompt=sample['prompt'],frame_prompts=sample['frame_prompts'],num_slots=16,positive=True)['context']
        cunit=WanVideoUnit_ImageEmbedderCLIP(); fresh_clip=cunit.process(pipe=pipe,input_image=input_image,end_image=None,height=inf['height'],width=inf['width'])['clip_feature']
        vunit=WanVideoUnit_ImageEmbedderVAE(); fresh_y=vunit.process(pipe=pipe,input_image=input_image,end_image=None,num_frames=num_frames,height=inf['height'],width=inf['width'],tiled=inf['tiled'],tile_size=tuple(inf['tile_size']),tile_stride=tuple(inf['tile_stride']))['y']
    text_cache=torch.load('/media/datasets/yumi/hjh/cache/aigbot_final/text_local/327/648642-685032/648649.pt',map_location='cpu',weights_only=False)
    vae_cache=torch.load('/media/datasets/yumi/hjh/cache/aigbot_final/vae/327/648642-685032/648649.pt',map_location='cpu',weights_only=False)
    comps=[tensor_compare('context',fresh_context,text_cache['context']), tensor_compare('clip_feature',fresh_clip,text_cache['clip_feature']), tensor_compare('y',fresh_y,vae_cache['y'])]
    (out/'cache_compare.json').write_text(json.dumps(comps,indent=2),encoding='utf-8')
    write_csv(out/'tables'/'cache_compare.csv',[{k:v for k,v in c.items() if k!='slot_cosines'} for c in comps])
    print('[cache_compare]', comps, flush=True)

    print('[stage] cfg sweep', flush=True)
    for scale in [1.0,2.0,3.0,5.0]:
        print(f'[cfg] {scale}', flush=True)
        frames=pipe(prompt=sample['prompt'], negative_prompt=sample['negative_prompt'], input_image=input_image, frame_prompts=sample['frame_prompts'], height=inf['height'], width=inf['width'], num_inference_steps=50, cfg_scale=scale, cfg_merge=inf['cfg_merge'], sigma_shift=inf['sigma_shift'], seed=args.seed, rand_device=inf['rand_device'], tiled=inf['tiled'], tile_size=tuple(inf['tile_size']), tile_stride=tuple(inf['tile_stride']), tea_cache_l1_thresh=inf.get('tea_cache_l1_thresh'), tea_cache_model_id=inf.get('tea_cache_model_id',''), framewise_decoding=inf.get('framewise_decoding',False), output_type=inf.get('output_type','quantized'))
        save_gt_pred_compare(sample, frames, out/'cfg_sweep', scale)
    print('[done]', out, flush=True)

if __name__=='__main__': main()
