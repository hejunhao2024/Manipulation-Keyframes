#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from einops import rearrange
from diffsynth.models.wan_video_dit import KeyframeCrossAttention, flash_attention
from keyframegen.train import agibot_non_ar_train as trainer

PARAM_GROUPS = (
    'self_attn_lora',
    'cross_attn_shared_lora',
    'cross_attn_text_lora',
    'cross_attn_image_lora',
    'norm3_full',
    'img_emb_full',
)
CROSS_PROJS = ['q', 'k', 'v', 'o', 'k_img', 'v_img']
SELF_PROJS = ['q', 'k', 'v', 'o']


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames=None):
    ensure_dir(path.parent)
    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_manifest(path: str):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line and not line.startswith('#'):
                out.append(line)
    return out


def sample_id_from_path(path: str):
    p = Path(path)
    return '/'.join(p.parts[-3:])


def task_from_path(path: str):
    return Path(path).parts[-3]


def select_samples(manifest: str, count: int):
    items = sorted(read_manifest(manifest), key=lambda x: sample_id_from_path(x))
    selected=[]; seen=set()
    for item in items:
        task=task_from_path(item)
        if task in seen:
            continue
        selected.append(item); seen.add(task)
        if len(selected) >= count:
            break
    return selected


def rms(x: torch.Tensor) -> float:
    return float(x.detach().float().pow(2).mean().sqrt().item())


def tensor_norm(x: torch.Tensor) -> float:
    return float(x.detach().float().norm().item())


def parameter_group_name(name: str) -> str:
    return trainer.classify_trainable_parameter(name)


def strict_load(pipe, cfg, checkpoint_path: str, rank: int = 0):
    groups, named_trainable = trainer.configure_trainable(pipe.dit, cfg['train'], rank=rank)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt['trainable_state_dict']
    params = dict(pipe.dit.named_parameters())
    expected = {name for name, p in pipe.dit.named_parameters() if p.requires_grad}
    state_keys = set(state)
    missing = sorted(expected - state_keys)
    unexpected = sorted(state_keys - expected)
    shape_mismatch = sorted(k for k in expected & state_keys if tuple(params[k].shape) != tuple(state[k].shape))
    summary = {
        'checkpoint_global_step': ckpt.get('global_step'),
        'checkpoint_loss': ckpt.get('loss'),
        'checkpoint_tensor_count': len(state),
        'expected_tensor_count': len(expected),
        'loaded_tensor_count': len(expected & state_keys) - len(shape_mismatch),
        'missing_keys': missing,
        'unexpected_keys': unexpected,
        'shape_mismatch_keys': shape_mismatch,
        'group_tensor_counts': {g: len(v) for g, v in groups.items()},
        'group_param_counts': {g: int(sum(p.numel() for _, p in v)) for g, v in groups.items()},
    }
    required = PARAM_GROUPS
    empty_required = [g for g in required if not groups.get(g)]
    if missing or unexpected or shape_mismatch or empty_required:
        summary['empty_required_groups'] = empty_required
        raise RuntimeError('Strict load failed: ' + json.dumps(summary, indent=2, default=str)[:4000])

    # Full-train deltas must be measured before loading.
    pre_load_full_rows = []
    for name, trained in state.items():
        if name not in params:
            continue
        group = parameter_group_name(name)
        if group not in ('norm3_full', 'img_emb_full'):
            continue
        base = params[name].detach().float().cpu()
        tr = trained.detach().float()
        delta = tr - base
        pre_load_full_rows.append({
            'name': name,
            'group': group,
            'module_type': group,
            'layer': layer_from_name(name),
            'proj': full_proj_from_name(name),
            'delta_norm': float(delta.norm().item()),
            'base_norm': float(base.norm().item()),
            'delta_to_base_ratio': float(delta.norm().item() / (base.norm().item() + 1e-12)),
        })

    pipe.dit.load_state_dict({k: v.to(dtype=params[k].dtype) for k, v in state.items()}, strict=False)
    pipe.dit.eval()
    return ckpt, state, groups, summary, pre_load_full_rows


def layer_from_name(name: str) -> int:
    parts = name.split('.')
    if len(parts) > 2 and parts[0] == 'blocks':
        try:
            return int(parts[1])
        except Exception:
            return -1
    return -1


def full_proj_from_name(name: str) -> str:
    if '.norm3.' in name:
        return 'norm3'
    if 'img_emb' in name:
        return 'img_emb'
    return name.split('.')[-2] if len(name.split('.')) > 1 else name


def lora_parameter_rows(pipe, full_rows):
    rows = []
    for name, module in pipe.dit.named_modules():
        if not isinstance(module, trainer.LoRALinear):
            continue
        base = module.base.weight.detach().float()
        a = module.lora_A.detach().float()
        b = module.lora_B.detach().float()
        gram_a = a @ a.t()
        gram_b = b.t() @ b
        delta_sq = torch.sum(gram_a * gram_b).clamp_min(0.0) * (module.scale ** 2)
        delta_norm = float(torch.sqrt(delta_sq).item())
        base_norm = float(base.norm().item())
        attn = 'self_attn' if '.self_attn.' in name else 'cross_attn' if '.cross_attn.' in name else 'other'
        proj = name.split('.')[-1]
        if attn == 'self_attn':
            group = 'self_attn_lora'
            module_type = f'self_attn.{proj}'
        elif proj in ('q', 'o'):
            group = 'cross_attn_shared_lora'
            module_type = f'cross_attn.{proj}'
        elif proj in ('k', 'v'):
            group = 'cross_attn_text_lora'
            module_type = f'cross_attn.{proj}'
        elif proj in ('k_img', 'v_img'):
            group = 'cross_attn_image_lora'
            module_type = f'cross_attn.{proj}'
        else:
            group = 'unknown_lora'
            module_type = f'{attn}.{proj}'
        rows.append({
            'name': name,
            'group': group,
            'module_type': module_type,
            'layer': layer_from_name(name),
            'proj': proj,
            'delta_norm': delta_norm,
            'base_norm': base_norm,
            'delta_to_base_ratio': delta_norm / (base_norm + 1e-12),
        })
    return rows + full_rows


def summarize_parameter_rows(rows):
    out=[]
    for key in sorted(set(r['module_type'] for r in rows)):
        vals=[float(r['delta_to_base_ratio']) for r in rows if r['module_type']==key]
        out.append({'module_type':key,'count':len(vals),'mean':float(np.mean(vals)),'median':float(np.median(vals)),'max':float(np.max(vals))})
    for key in sorted(set(r['group'] for r in rows)):
        vals=[float(r['delta_to_base_ratio']) for r in rows if r['group']==key]
        out.append({'module_type':'GROUP:'+key,'count':len(vals),'mean':float(np.mean(vals)),'median':float(np.median(vals)),'max':float(np.max(vals))})
    return out



def heat_color(value, vmax):
    t=0.0 if vmax<=0 or value!=value else max(0.0,min(1.0,value/vmax))
    return (int(255*t), int(220*(1-abs(t-0.5)*2)), int(255*(1-t)))


def plot_parameter_heatmap(rows, out_path):
    mat=np.full((40,len(CROSS_PROJS)), np.nan)
    for r in rows:
        if not str(r['module_type']).startswith('cross_attn.'):
            continue
        layer=int(r['layer']); proj=r['proj']
        if 0 <= layer < 40 and proj in CROSS_PROJS:
            mat[layer, CROSS_PROJS.index(proj)] = float(r['delta_to_base_ratio'])
    cell_w, cell_h = 72, 18
    left, top = 70, 34
    vmax=float(np.nanmax(mat)) if np.isfinite(mat).any() else 1.0
    img=Image.new('RGB',(left+cell_w*len(CROSS_PROJS)+20, top+cell_h*40+30),'white')
    d=ImageDraw.Draw(img)
    d.text((8,6),f'cross-attn delta/base vmax={vmax:.4g}',fill='black')
    for j,p in enumerate(CROSS_PROJS): d.text((left+j*cell_w+4, top-18), p, fill='black')
    for i in range(40):
        d.text((8,top+i*cell_h+3),f'layer {i:02d}',fill='black')
        for j in range(len(CROSS_PROJS)):
            val=mat[i,j]
            color=(230,230,230) if not np.isfinite(val) else heat_color(float(val),vmax)
            x0=left+j*cell_w; y0=top+i*cell_h
            d.rectangle([x0,y0,x0+cell_w-2,y0+cell_h-2],fill=color)
    img.save(out_path)


def simple_bar(labels, values, path, title, ylabel='value'):
    w=max(520, 80*len(labels)+80); h=360
    img=Image.new('RGB',(w,h),'white'); d=ImageDraw.Draw(img)
    d.text((10,8),title,fill='black'); d.text((10,28),ylabel,fill='black')
    vmax=max([abs(float(v)) for v in values]+[1e-12])
    zero_y=260
    bar_w=max(12,(w-120)//max(1,len(labels))-8)
    for i,(lab,val) in enumerate(zip(labels,values)):
        x=70+i*(bar_w+8); y=zero_y-int(float(val)/vmax*180)
        color=(70,120,220) if val>=0 else (220,80,80)
        d.rectangle([x,min(y,zero_y),x+bar_w,max(y,zero_y)],fill=color)
        d.text((x, zero_y+6), str(lab)[:16], fill='black')
        d.text((x, min(y,zero_y)-14), f'{float(val):.2g}', fill='black')
    img.save(path)


def plot_group_bars(summary, out_path):
    groups=[r for r in summary if r['module_type'].startswith('GROUP:')]
    labels=[r['module_type'].replace('GROUP:','') for r in groups]
    simple_bar(labels,[r['mean'] for r in groups],out_path,'parameter delta by group','mean delta/base')

def load_items(sample_dirs, cfg, device):
    data_cfg=cfg['data']; expected=int(data_cfg['expected_num_slots'])
    items=[]
    for d in sample_dirs:
        item=trainer.load_cached_item(d, data_cfg, cfg['train']['conditioning_mode'], device, expected)
        items.append(item)
    return items


def make_prompt_variants(items, seed):
    by_task={item['sample_id'].split('/',1)[0]: item for item in items}
    variants={}
    g=torch.Generator(device='cpu').manual_seed(seed)
    for item in items:
        ctx=item['conditions']['context']
        sid=item['sample_id']; task=sid.split('/',1)[0]
        wrong=None
        for other in items:
            if other['sample_id'].split('/',1)[0] != task:
                wrong=other; break
        if wrong is None:
            wrong=items[(items.index(item)+1)%len(items)]
        perm=torch.randperm(ctx.shape[1], generator=g)
        variants[sid]={
            'correct': ctx,
            'empty': torch.zeros_like(ctx),
            'wrong': wrong['conditions']['context'].to(ctx),
            'shuffled': ctx[:, perm].contiguous(),
            'wrong_item': wrong,
        }
    return variants


def make_noisy(target, sigma, seed):
    gen=torch.Generator(device='cpu').manual_seed(seed)
    noise=torch.randn(target.shape, generator=gen, dtype=torch.float32).to(target)
    noisy=(1.0-sigma)*target + sigma*noise
    velocity=noise-target
    timestep=torch.tensor([sigma*1000.0], device=target.device, dtype=target.dtype)
    return noisy, velocity, timestep


def loss_parts(pred, velocity):
    err=(pred.float()-velocity.float()).pow(2)
    return err.mean(), err


def per_slot_norm(x, slots):
    # x [B,C,F,H,W]
    vals=[]
    for s in slots:
        vals.append(float(x[:,:,s].float().norm().item()))
    return vals


def forward_model(pipe, item, context, noisy, timestep, branch_mode='normal', activation=None, lora_activation=None):
    with patch_lora_for_activation(lora_activation), patch_cross_attention(branch_mode, activation):
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=noisy.is_cuda):
            return pipe.model_fn(
                dit=pipe.dit,
                latents=noisy,
                timestep=timestep.to(noisy),
                context=context.to(noisy.device),
                clip_feature=item['clip_feature'],
                y=item['y'],
                fuse_vae_embedding_in_latents=item['fuse_vae_embedding_in_latents'],
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )


@contextmanager
def patch_lora_for_activation(store):
    if store is None:
        yield; return
    originals=[]
    def make_forward(module_name, module):
        def fwd(self, x):
            base_out=self.base(x)
            z=self.dropout(x).to(self.lora_A)
            z=z @ self.lora_A.t()
            z=z @ self.lora_B.t()
            lora_out=z.to(base_out) * self.scale
            attn='self_attn' if '.self_attn.' in module_name else 'cross_attn' if '.cross_attn.' in module_name else 'other'
            proj=module_name.split('.')[-1]
            if attn == 'cross_attn' and proj in ('k_img','v_img'):
                group='cross_attn_image_lora'
            elif attn == 'cross_attn' and proj in ('k','v'):
                group='cross_attn_text_lora'
            elif attn == 'cross_attn':
                group='cross_attn_shared_lora'
            elif attn == 'self_attn':
                group='self_attn_lora'
            else:
                group='other'
            store.append({'name':module_name,'group':group,'layer':layer_from_name(module_name),'proj':proj,'base_output_rms':rms(base_out),'lora_output_rms':rms(lora_out),'lora_to_base_ratio':rms(lora_out)/(rms(base_out)+1e-12)})
            return base_out + lora_out
        return fwd
    try:
        # monkeypatch only modules already instantiated in current global model via stack inspection not possible here.
        # Caller sets _diagnostic_lora_modules on function.
        mods=getattr(patch_lora_for_activation, '_mods', [])
        for name, module in mods:
            originals.append((module, module.forward))
            module.forward=MethodType(make_forward(name, module), module)
        yield
    finally:
        for module, orig in originals:
            module.forward=orig


@contextmanager
def patch_cross_attention(mode='normal', store=None):
    originals=[]
    def fwd(self, x, y):
        b,n,c=x.shape
        residual_rms=rms(x)
        if y.dim()==3:
            f=y.shape[0]; y=y.unsqueeze(0).expand(b,-1,-1,-1)
        elif y.dim()==4:
            f=y.shape[1]
        else:
            raise ValueError(f'bad context shape {y.shape}')
        hw=n//f
        xx=rearrange(x, 'b (f hw) c -> (b f) hw c', f=f, hw=hw)
        yy=rearrange(y, 'b f l c -> (b f) l c')
        if self.has_image_input:
            img=yy[:,:257]; ctx=yy[:,257:]
        else:
            img=None; ctx=yy
        q=self.norm_q(self.q(xx))
        k=self.norm_k(self.k(ctx)); v=self.v(ctx)
        text_out=self.attn(q,k,v)
        if mode=='no_cross_text':
            text_out=torch.zeros_like(text_out)
        img_out=None
        xsum=text_out
        if self.has_image_input:
            k_img=self.norm_k_img(self.k_img(img)); v_img=self.v_img(img)
            img_out=flash_attention(q,k_img,v_img,num_heads=self.num_heads)
            if mode=='no_cross_image':
                img_out=torch.zeros_like(img_out)
            xsum=xsum+img_out
        if store is not None:
            layer=getattr(self, '_diagnostic_layer', -1)
            text_r=rms(text_out)
            img_r=rms(img_out) if img_out is not None else 0.0
            store.append({'layer':layer,'text_branch_output_rms':text_r,'image_branch_output_rms':img_r,'image_to_text_output_ratio':img_r/(text_r+1e-12),'cross_output_rms':rms(xsum),'residual_input_rms':residual_rms,'mode':mode})
        out=rearrange(xsum, '(b f) hw c -> b (f hw) c', b=b, f=f, hw=hw)
        return self.o(out)
    try:
        mods=getattr(patch_cross_attention, '_mods', [])
        for layer, module in mods:
            setattr(module, '_diagnostic_layer', layer)
            originals.append((module, module.forward))
            module.forward=MethodType(fwd, module)
        yield
    finally:
        for module, orig in originals:
            module.forward=orig


def setup_patch_module_lists(pipe):
    lora=[]; cross=[]
    for name, module in pipe.dit.named_modules():
        if isinstance(module, trainer.LoRALinear):
            lora.append((name,module))
        if isinstance(module, KeyframeCrossAttention):
            cross.append((layer_from_name(name), module))
    patch_lora_for_activation._mods=lora
    patch_cross_attention._mods=cross


def aggregate_activation(rows):
    out=[]
    if not rows:
        return out
    for layer in sorted(set(r['layer'] for r in rows)):
        ss=[r for r in rows if r['layer']==layer]
        out.append({'layer':layer,'text_branch_output_rms':float(np.mean([r['text_branch_output_rms'] for r in ss])),'image_branch_output_rms':float(np.mean([r['image_branch_output_rms'] for r in ss])),'image_to_text_output_ratio':float(np.mean([r['image_to_text_output_ratio'] for r in ss])),'cross_output_rms':float(np.mean([r['cross_output_rms'] for r in ss])),'residual_input_rms':float(np.mean([r['residual_input_rms'] for r in ss]))})
    return out


def aggregate_lora_activation(rows):
    out=[]
    for group in sorted(set(r['group'] for r in rows)):
        ss=[r for r in rows if r['group']==group]
        out.append({'group':group,'count':len(ss),'base_output_rms':float(np.mean([r['base_output_rms'] for r in ss])),'lora_output_rms':float(np.mean([r['lora_output_rms'] for r in ss])),'lora_to_base_ratio':float(np.mean([r['lora_to_base_ratio'] for r in ss]))})
    return out



def simple_lines(xs, series, labels, path, title, ylabel='RMS'):
    w,h=760,360; left,top,right,bottom=55,35,20,55
    img=Image.new('RGB',(w,h),'white'); d=ImageDraw.Draw(img)
    d.text((10,8),title,fill='black'); d.text((10,24),ylabel,fill='black')
    all_vals=[float(v) for vals in series for v in vals]
    vmax=max(all_vals+[1e-12]); vmin=min(all_vals+[0.0])
    if abs(vmax-vmin)<1e-12: vmax=vmin+1.0
    def xy(i,val):
        x=left + (w-left-right) * (i/max(1,len(xs)-1))
        y=h-bottom - (h-top-bottom) * ((float(val)-vmin)/(vmax-vmin))
        return int(x), int(y)
    colors=[(40,90,220),(220,80,50),(40,160,90)]
    for si,vals in enumerate(series):
        pts=[xy(i,v) for i,v in enumerate(vals)]
        for a,b in zip(pts,pts[1:]): d.line([a,b],fill=colors[si%len(colors)],width=2)
        d.text((w-right-150, top+16*si), labels[si], fill=colors[si%len(colors)])
    d.line([(left,h-bottom),(w-right,h-bottom)],fill='black')
    d.line([(left,top),(left,h-bottom)],fill='black')
    img.save(path)


def plot_activations(layer_rows, lora_summary, fig_dir):
    if layer_rows:
        x=[r['layer'] for r in layer_rows]
        simple_lines(x, [[r['text_branch_output_rms'] for r in layer_rows], [r['image_branch_output_rms'] for r in layer_rows]], ['text','image'], fig_dir/'text_image_branch_rms_by_layer.png', 'text/image branch RMS by layer')
        simple_lines(x, [[r['image_to_text_output_ratio'] for r in layer_rows]], ['image/text'], fig_dir/'image_text_ratio_by_layer.png', 'image/text RMS ratio by layer', 'ratio')
    if lora_summary:
        labels=[r['group'] for r in lora_summary]
        simple_bar(labels,[r['lora_to_base_ratio'] for r in lora_summary],fig_dir/'lora_activation_by_group.png','LoRA activation by group','lora/base RMS')

def run_diagnostics(args):
    t0=time.time(); stage_times={}; forward_count=0
    out=Path(args.output_dir); table_dir=out/'tables'; fig_dir=out/'figures'; matrix_dir=out/'matrices'
    for d in (out,table_dir,fig_dir,matrix_dir): ensure_dir(d)
    cfg=load_json(args.config)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda': torch.cuda.set_device(0)

    selected=select_samples(args.manifest,args.sample_count)
    (out/'selected_samples.txt').write_text('\n'.join(selected)+'\n', encoding='utf-8')
    print('[selected]', selected, flush=True)

    t=time.time(); print('[stage] build/load model', flush=True)
    pipe=trainer.build_pipe(cfg, cfg['train']['conditioning_mode'], local_rank=0)
    pipe.load_models_to_device(['dit'])
    ckpt,state,groups,load_summary,full_rows = strict_load(pipe,cfg,args.checkpoint,rank=0)
    setup_patch_module_lists(pipe)
    stage_times['load_model']=time.time()-t
    (out/'checkpoint_load_summary.json').write_text(json.dumps(load_summary, indent=2), encoding='utf-8')

    t=time.time(); print('[stage] parameter delta', flush=True)
    param_rows=lora_parameter_rows(pipe, full_rows)
    param_summary=summarize_parameter_rows(param_rows)
    write_csv(table_dir/'parameter_delta_long.csv', param_rows)
    write_csv(table_dir/'parameter_delta_summary.csv', param_summary)
    plot_parameter_heatmap(param_rows, fig_dir/'cross_attn_delta_heatmap.png')
    plot_group_bars(param_summary, fig_dir/'parameter_delta_by_group.png')
    stage_times['parameter_delta']=time.time()-t

    t=time.time(); print('[stage] load cached samples', flush=True)
    items=load_items(selected,cfg,pipe.device)
    variants=make_prompt_variants(items,args.seed)
    stage_times['load_cache']=time.time()-t

    correct_cache={}; activation_rows=[]; lora_activation_rows=[]; prompt_rows=[]
    prompt_sigmas=[float(x) for x in args.prompt_sigmas]
    t=time.time(); print('[stage] prompt causal + activation on correct sigma=0.5', flush=True)
    for sample_idx,item in enumerate(items):
        sid=item['sample_id']
        for sigma in prompt_sigmas:
            noisy,velocity,timestep=make_noisy(item['target_latents'], sigma, args.seed + sample_idx*1000 + int(sigma*100))
            act_store=[] if abs(sigma-0.5)<1e-6 else None
            lora_store=[] if abs(sigma-0.5)<1e-6 else None
            pred=forward_model(pipe,item,variants[sid]['correct'],noisy,timestep,'normal',act_store,lora_store)
            forward_count+=1
            loss,err=loss_parts(pred,velocity)
            correct_cache[(sid,sigma)]={'pred':pred.detach(), 'loss':loss.detach(), 'velocity':velocity, 'noisy':noisy, 'timestep':timestep, 'err':err.detach()}
            if act_store is not None:
                for r in act_store: r.update({'sample_id':sid,'sigma':sigma}); activation_rows.extend(act_store)
                for r in lora_store: r.update({'sample_id':sid,'sigma':sigma}); lora_activation_rows.extend(lora_store)
            correct_norm=pred.float().norm().clamp_min(1e-12)
            for var in ('correct','empty','wrong','shuffled'):
                if var=='correct':
                    pv=pred; lv=loss; ev=err
                else:
                    pv=forward_model(pipe,item,variants[sid][var],noisy,timestep)
                    forward_count+=1
                    lv,ev=loss_parts(pv,velocity)
                diff=(pv-correct_cache[(sid,sigma)]['pred']).float()
                row={'sample_id':sid,'task':sid.split('/',1)[0],'sigma':sigma,'variant':var,'sensitivity':float(diff.norm().item()/(float(correct_norm.item())+1e-12)),'loss_correct':float(loss.item()),'loss_variant':float(lv.item()),'delta_loss':float((lv-loss).item()),'correct_better':bool((lv-loss).item()>0)}
                for slot in range(item['target_latents'].shape[2]):
                    denom=pred[:,:,slot].float().norm().clamp_min(1e-12)
                    row[f'slot_{slot}_sensitivity']=float(diff[:,:,slot].float().norm().item()/(float(denom.item())+1e-12))
                prompt_rows.append(row)
                if var!='correct': del pv
    write_csv(table_dir/'prompt_causal_long.csv', prompt_rows)
    prompt_summary=[]
    for sigma in prompt_sigmas:
        for var in ('empty','wrong','shuffled'):
            ss=[r for r in prompt_rows if r['sigma']==sigma and r['variant']==var]
            prompt_summary.append({'variant':var,'sigma':sigma,'mean_sensitivity':float(np.mean([r['sensitivity'] for r in ss])),'mean_delta_loss':float(np.mean([r['delta_loss'] for r in ss])),'median_delta_loss':float(np.median([r['delta_loss'] for r in ss])),'correct_better_fraction':float(np.mean([1.0 if r['correct_better'] else 0.0 for r in ss])),'num_samples':len(ss)})
    write_csv(table_dir/'prompt_causal_summary.csv', prompt_summary)
    plot_prompt(prompt_summary, fig_dir)
    stage_times['prompt_activation']=time.time()-t

    t=time.time(); print('[stage] activation summaries', flush=True)
    layer_act=aggregate_activation(activation_rows); lora_act=aggregate_lora_activation(lora_activation_rows)
    write_csv(table_dir/'activation_by_layer.csv', layer_act)
    write_csv(table_dir/'activation_summary.csv', lora_act)
    plot_activations(layer_act,lora_act,fig_dir)
    stage_times['activation_summary']=time.time()-t

    slots=[int(x) for x in args.intervention_slots]
    t=time.time(); print('[stage] 5x5 slot intervention', flush=True)
    mat_sum=np.zeros((len(slots),len(slots)),dtype=np.float64); slot_rows=[]
    for sample_idx,item in enumerate(items):
        sid=item['sample_id']; cache=correct_cache[(sid,0.5)]; pred0=cache['pred']; noisy=cache['noisy']; timestep=cache['timestep']
        wrong_ctx=variants[sid]['wrong'].to(variants[sid]['correct'])
        for j_idx,j in enumerate(slots):
            edited=variants[sid]['correct'].clone(); edited[:,j]=wrong_ctx[:,j]
            pred=forward_model(pipe,item,edited,noisy,timestep); forward_count+=1
            for i_idx,i in enumerate(slots):
                val=float((pred[:,:,i]-pred0[:,:,i]).float().norm().item()/(float(pred0[:,:,i].float().norm().item())+1e-12))
                mat_sum[i_idx,j_idx]+=val
                slot_rows.append({'sample_id':sid,'output_slot':i,'prompt_slot':j,'sensitivity':val,'on_diagonal':i==j})
            del pred, edited
    mat=mat_sum/len(items)
    np.save(matrix_dir/'slot_intervention_5x5.npy', mat)
    write_csv(table_dir/'slot_intervention_5x5_long.csv', slot_rows)
    diag=np.diag(mat); off=mat[~np.eye(len(slots),dtype=bool)]
    argmax_diag=np.mean([1.0 if slots[int(np.argmax(mat[i]))]==slots[i] else 0.0 for i in range(len(slots))])
    slot_summary=[{'diagonal_mean':float(diag.mean()),'off_diagonal_mean':float(off.mean()),'diag_off_ratio':float(diag.mean()/(off.mean()+1e-12)),'diagonal_fraction':float(diag.sum()/(mat.sum()+1e-12)),'argmax_on_diagonal_fraction':float(argmax_diag),'random_argmax_baseline':1.0/len(slots)}]
    write_csv(table_dir/'slot_intervention_5x5_summary.csv', slot_summary)
    plot_slot_matrix(mat, slots, fig_dir/'slot_intervention_5x5.png')
    stage_times['slot_intervention']=time.time()-t

    t=time.time(); print('[stage] branch ablation', flush=True)
    ablation_rows=[]
    for item in items:
        sid=item['sample_id']; cache=correct_cache[(sid,0.5)]; pred0=cache['pred']; loss0=cache['loss']; noisy=cache['noisy']; timestep=cache['timestep']; velocity=cache['velocity']
        norm0=pred0.float().norm().clamp_min(1e-12)
        for mode in ('no_cross_text','no_cross_image'):
            pred=forward_model(pipe,item,variants[sid]['correct'],noisy,timestep,branch_mode=mode); forward_count+=1
            loss,err=loss_parts(pred,velocity)
            row={'sample_id':sid,'mode':mode,'sensitivity':float((pred-pred0).float().norm().item()/(float(norm0.item())+1e-12)),'loss_normal':float(loss0.item()),'loss':float(loss.item()),'loss_increase':float((loss-loss0).item())}
            for s in range(item['target_latents'].shape[2]):
                row[f'slot_{s}_loss_increase']=float((err[:,:,s].mean()-cache['err'][:,:,s].mean()).item())
            ablation_rows.append(row); del pred
    write_csv(table_dir/'branch_ablation_long.csv', ablation_rows)
    ab_summary=[]
    for mode in ('no_cross_text','no_cross_image'):
        ss=[r for r in ablation_rows if r['mode']==mode]
        ab_summary.append({'mode':mode,'mean_sensitivity':float(np.mean([r['sensitivity'] for r in ss])),'mean_loss_increase':float(np.mean([r['loss_increase'] for r in ss])),'median_loss_increase':float(np.median([r['loss_increase'] for r in ss])),'positive_fraction':float(np.mean([1.0 if r['loss_increase']>0 else 0.0 for r in ss])),'num_samples':len(ss)})
    write_csv(table_dir/'branch_ablation_summary.csv', ab_summary)
    plot_branch(ab_summary, fig_dir/'branch_ablation_loss_increase.png')
    stage_times['branch_ablation']=time.time()-t

    total=time.time()-t0
    peak= torch.cuda.max_memory_allocated(0)/1024**3 if torch.cuda.is_available() else 0.0
    summary=make_summary(load_summary,param_summary,layer_act,lora_act,prompt_summary,slot_summary[0],ab_summary,forward_count,stage_times,total,peak)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    write_csv(table_dir/'diagnosis_summary.csv', flatten_summary(summary))
    write_report(out/'report.md', summary, load_summary, param_summary, layer_act, lora_act, prompt_summary, slot_summary[0], ab_summary)
    print(f'[done] {out} forwards={forward_count} time={total:.1f}s peak_gb={peak:.2f}', flush=True)



def plot_prompt(rows, fig_dir):
    labels=[f"{r['variant']}@{r['sigma']}" for r in rows]
    simple_bar(labels,[r['mean_delta_loss'] for r in rows],fig_dir/'prompt_causal_summary.png','prompt causal delta_loss','mean delta_loss')
    simple_bar(labels,[r['correct_better_fraction'] for r in rows],fig_dir/'prompt_correct_better_fraction.png','prompt correct better fraction','fraction')


def plot_slot_matrix(mat, slots, path):
    cell=54; left=70; top=42
    vmax=float(np.max(mat)) if mat.size else 1.0
    img=Image.new('RGB',(left+cell*len(slots)+25, top+cell*len(slots)+35),'white')
    d=ImageDraw.Draw(img)
    d.text((10,8),f'5x5 slot intervention vmax={vmax:.4g}',fill='black')
    for j,s in enumerate(slots): d.text((left+j*cell+16,top-20),str(s),fill='black')
    for i,s in enumerate(slots):
        d.text((18,top+i*cell+18),str(s),fill='black')
        for j in range(len(slots)):
            val=float(mat[i,j]); x=left+j*cell; y=top+i*cell
            d.rectangle([x,y,x+cell-2,y+cell-2],fill=heat_color(val,vmax))
            d.text((x+5,y+18),f'{val:.2g}',fill='black')
    img.save(path)


def plot_branch(rows, path):
    simple_bar([r['mode'] for r in rows],[r['mean_loss_increase'] for r in rows],path,'branch ablation loss increase','mean loss increase')

def get_summary_value(param_summary, module_type):
    for r in param_summary:
        if r['module_type']==module_type:
            return r
    return {'mean':0,'median':0,'max':0,'count':0}


def make_summary(load_summary,param_summary,layer_act,lora_act,prompt_summary,slot_summary,ab_summary,forward_count,stage_times,total,peak):
    vimg=get_summary_value(param_summary,'cross_attn.v_img')
    vtxt=get_summary_value(param_summary,'cross_attn.v')
    kimg=get_summary_value(param_summary,'cross_attn.k_img')
    ktxt=get_summary_value(param_summary,'cross_attn.k')
    max_mod=max([r for r in param_summary if not r['module_type'].startswith('GROUP:')], key=lambda r:r['mean']) if param_summary else {}
    act_ratio=float(np.mean([r['image_to_text_output_ratio'] for r in layer_act])) if layer_act else None
    text_r=float(np.mean([r['text_branch_output_rms'] for r in layer_act])) if layer_act else None
    img_r=float(np.mean([r['image_branch_output_rms'] for r in layer_act])) if layer_act else None
    wrong=[r for r in prompt_summary if r['variant']=='wrong']
    shuffled=[r for r in prompt_summary if r['variant']=='shuffled']
    no_text=next((r for r in ab_summary if r['mode']=='no_cross_text'),{})
    no_img=next((r for r in ab_summary if r['mode']=='no_cross_image'),{})
    red=[]
    if wrong and np.mean([r['mean_delta_loss'] for r in wrong]) <= 0: red.append('wrong_mean_delta_loss_nonpositive')
    if shuffled and np.mean([r['mean_delta_loss'] for r in shuffled]) <= 0: red.append('shuffled_mean_delta_loss_nonpositive')
    if wrong and np.mean([r['correct_better_fraction'] for r in wrong]) <= 0.5: red.append('wrong_correct_better_fraction_le_0p5')
    if slot_summary['diag_off_ratio'] <= 1.2: red.append('slot_diag_off_ratio_near_1')
    if slot_summary['argmax_on_diagonal_fraction'] <= 0.25: red.append('slot_argmax_near_random')
    if no_text and no_text.get('mean_loss_increase',0) <= 0: red.append('no_cross_text_loss_increase_nonpositive')
    if act_ratio is not None and act_ratio > 5: red.append('image_branch_activation_much_larger_than_text')
    likely=[]
    if vimg['mean'] > max(vtxt['mean'],1e-12)*2: likely.append('parameter_adaptation_imbalance')
    if act_ratio is not None and act_ratio > 3: likely.append('activation_imbalance')
    if red.count('wrong_mean_delta_loss_nonpositive') or red.count('shuffled_mean_delta_loss_nonpositive') or (wrong and np.mean([r['correct_better_fraction'] for r in wrong]) <= 0.5): likely.append('weak_prompt_semantics')
    if slot_summary['diag_off_ratio'] <= 1.3: likely.append('weak_slot_locality')
    if ('weak_prompt_semantics' in likely or 'weak_slot_locality' in likely) and no_text and no_text.get('mean_loss_increase',0) <= 0: likely.append('task-template_shortcut')
    if not likely: likely=['no obvious issue']
    return {'forward_count':forward_count,'stage_times_sec':stage_times,'total_time_sec':total,'peak_memory_gb':peak,'checkpoint':load_summary,'parameter':{'v_img_mean_delta_base':vimg['mean'],'text_v_mean_delta_base':vtxt['mean'],'v_img_over_text_v':vimg['mean']/(vtxt['mean']+1e-12),'k_img_over_text_k':kimg['mean']/(ktxt['mean']+1e-12),'largest_mean_module':max_mod,'delta_gt_1_count':None},'activation':{'mean_text_branch_rms':text_r,'mean_image_branch_rms':img_r,'mean_image_to_text_ratio':act_ratio},'slot_intervention':slot_summary,'branch_ablation':ab_summary,'prompt_causal':prompt_summary,'red_flags':red,'final_diagnosis':likely}


def flatten_summary(summary):
    rows=[]
    for k,v in summary.items():
        if isinstance(v,(str,int,float)) or v is None:
            rows.append({'key':k,'value':v})
        elif isinstance(v,dict):
            for kk,vv in v.items():
                rows.append({'key':f'{k}.{kk}','value':json.dumps(vv) if isinstance(vv,(dict,list)) else vv})
        else:
            rows.append({'key':k,'value':json.dumps(v)})
    return rows


def write_report(path, summary, load_summary, param_summary, layer_act, lora_act, prompt_summary, slot_summary, ab_summary):
    p=summary['parameter']; a=summary['activation']
    lines=['# Exp5b Step 1000 Fast Condition Diagnosis','', '## 1. Checkpoint verification','', f"global_step: {load_summary['checkpoint_global_step']}", f"checkpoint tensors: {load_summary['checkpoint_tensor_count']}", f"loaded tensors: {load_summary['loaded_tensor_count']}", f"missing keys: {len(load_summary['missing_keys'])}", f"unexpected keys: {len(load_summary['unexpected_keys'])}", f"shape mismatches: {len(load_summary['shape_mismatch_keys'])}", '', '## 2. Parameter adaptation','', f"v_img mean delta/base: {p['v_img_mean_delta_base']:.6g}", f"text_v mean delta/base: {p['text_v_mean_delta_base']:.6g}", f"v_img / text_v: {p['v_img_over_text_v']:.6g}", f"k_img / text_k: {p['k_img_over_text_k']:.6g}", f"largest mean module: {p['largest_mean_module']}", '', '## 3. Real forward activations','', f"mean text branch RMS: {a['mean_text_branch_rms']}", f"mean image branch RMS: {a['mean_image_branch_rms']}", f"mean image/text ratio: {a['mean_image_to_text_ratio']}", '', '## 4. Prompt causal probe','', '| variant | sigma | sensitivity | delta_loss | correct_better_fraction |', '|---|---:|---:|---:|---:|']
    for r in prompt_summary:
        lines.append(f"| {r['variant']} | {r['sigma']} | {r['mean_sensitivity']:.6g} | {r['mean_delta_loss']:.6g} | {r['correct_better_fraction']:.3f} |")
    lines += ['', '## 5. 5x5 slot intervention','', f"diag_mean: {slot_summary['diagonal_mean']:.6g}", f"off_diag_mean: {slot_summary['off_diagonal_mean']:.6g}", f"diag/off ratio: {slot_summary['diag_off_ratio']:.6g}", f"argmax_on_diagonal_fraction: {slot_summary['argmax_on_diagonal_fraction']:.3f}", '', '## 6. Text/image branch ablation','', '| mode | sensitivity | loss_increase | positive_fraction |', '|---|---:|---:|---:|']
    for r in ab_summary:
        lines.append(f"| {r['mode']} | {r['mean_sensitivity']:.6g} | {r['mean_loss_increase']:.6g} | {r['positive_fraction']:.3f} |")
    lines += ['', '## 7. Final diagnosis','', f"red_flags: {', '.join(summary['red_flags']) if summary['red_flags'] else 'none'}", f"most_likely: {', '.join(summary['final_diagnosis'])}", f"forwards: {summary['forward_count']}", f"total_time_sec: {summary['total_time_sec']:.1f}", f"peak_memory_gb: {summary['peak_memory_gb']:.2f}"]
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--sample-count', type=int, default=4)
    ap.add_argument('--prompt-sigmas', type=float, nargs='+', default=[0.5,0.8])
    ap.add_argument('--intervention-slots', type=int, nargs='+', default=[1,4,8,12,15])
    ap.add_argument('--seed', type=int, default=20260716)
    args=ap.parse_args()
    run_diagnostics(args)

if __name__ == '__main__':
    main()
