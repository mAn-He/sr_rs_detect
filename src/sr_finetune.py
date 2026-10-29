"""S6: Real-ESRGAN 공식 paired data 파인튜닝(docs/Training.md). 공식 finetune_realesrgan_x4plus_pairdata.yml을 복사해
pretrain_network_g를 RealESRGAN_x4plus.pth, netD를 RealESRGAN_x4plus_netD.pth로 두고 데이터 경로와 반복 횟수만 바꾼다.
학습 쌍은 train 블록의 실제 HR(GT)·LR(LQ) 타일이며 학습 직전 tiles_cache로 복사하고 끝나면 삭제한다."""
import json
import math
import shutil
import time
from pathlib import Path

import torch
import yaml

import gdal_ops as g
from sr_run import TP, WT, official

RE = TP / 'Real-ESRGAN'
name = 'S6_realesrgan_ft_spacenet' + ('_smoke' if g.CFG['smoke'] else '')
cache = Path(g.CFG['tiles_cache']) / 's6'
shutil.rmtree(cache, ignore_errors=True)
shutil.copytree(g.W / 'tiles/hr/images/train', cache / 'gt')
shutil.copytree(g.W / 'tiles/lr/images/train', cache / 'lq')
official('official_sr', ['scripts/generate_meta_info_pairdata.py', '--input', cache / 'gt', cache / 'lq',
                         '--root', cache / 'gt', cache / 'lq', '--meta_info', cache / 'meta_info.txt'], RE)
opt = yaml.safe_load((RE / 'options/finetune_realesrgan_x4plus_pairdata.yml').read_text())
opt['name'] = name
ds = opt['datasets']['train']
ds['dataroot_gt'], ds['dataroot_lq'], ds['meta_info'] = str(cache / 'gt'), str(cache / 'lq'), str(cache / 'meta_info.txt')
opt['path']['pretrain_network_g'] = str(WT / 'realesrgan/RealESRGAN_x4plus.pth')
opt['path']['pretrain_network_d'] = str(WT / 'realesrgan/RealESRGAN_x4plus_netD.pth')
n_pairs = len(list((cache / 'gt').iterdir()))
if g.CFG['smoke']:
    # BasicSR 로더는 drop_last라 쌍 수가 배치(12)보다 적으면 배치가 0개가 되어 학습이 되지 않는다.
    # 스모크에서만 공식 옵션 dataset_enlarge_ratio로 쌍을 반복해 배치 1개 이상을 만든다.
    ds['dataset_enlarge_ratio'] = math.ceil(ds['batch_size_per_gpu'] / n_pairs)
assert n_pairs * ds['dataset_enlarge_ratio'] >= ds['batch_size_per_gpu'], (n_pairs, ds['batch_size_per_gpu'])
iters = g.CFG['s6_smoke_iter'] if g.CFG['smoke'] else g.CFG['s6_total_iter']
opt['train']['total_iter'] = iters
# 로그 간격만 반복 횟수에 맞춘다(학습 설정 아님): 짧은 스모크에서도 반복당 시간이 로그에 남게 한다.
opt['logger']['print_freq'] = min(opt['logger']['print_freq'], max(iters // 4, 1))
yml = cache / f'{name}.yml'
yml.write_text(yaml.safe_dump(opt, sort_keys=False))
shutil.copy(yml, g.ROOT / 'results' / f'{name}.yml')
t0 = time.time()
official('official_sr', ['realesrgan/train.py', '-opt', yml, '--auto_resume'], RE)
sec = time.time() - t0
net = RE / 'experiments' / name / 'models/net_g_latest.pth'
# 학습이 실제로 가중치를 바꿨는지 확인한다(바뀌지 않으면 S6 = S4).
pre = torch.load(WT / 'realesrgan/RealESRGAN_x4plus.pth', map_location='cpu')['params_ema']
post = torch.load(net, map_location='cpu')['params']
delta = max((post[k].float() - pre[k].float()).abs().max().item() for k in pre)
assert delta > 0, f'S6 weights unchanged after {iters} iterations'
info = {'name': name, 'total_iter': iters, 'n_pairs': n_pairs, 'dataset_enlarge_ratio': ds['dataset_enlarge_ratio'],
        'max_abs_weight_change': delta, 'seconds': round(sec, 1),
        'batch_size_per_gpu': ds['batch_size_per_gpu'], 'gt_size': ds['gt_size'], 'net_g': str(net),
        'train_log': str(max((RE / 'experiments' / name).glob('train_*.log')))}
(g.W / 's6_finetune.json').write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
shutil.rmtree(cache)
