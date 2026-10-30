"""S6: Real-ESRGAN 공식 paired data 파인튜닝(docs/Training.md). 공식 finetune_realesrgan_x4plus_pairdata.yml을 복사해
pretrain_network_g를 RealESRGAN_x4plus.pth, netD를 RealESRGAN_x4plus_netD.pth로 두고 데이터 경로, 반복 횟수, milestones만 바꾼다.
학습 쌍은 train 블록의 실제 HR(GT)·LR(LQ) 타일이며 학습 직전 tiles_cache로 복사하고 끝나면 삭제한다.
반복 횟수(전체 실행): 같은 데이터로 s6_measure_iter 반복을 돌려 로그 시각으로 반복당 시간을 재고,
s6_budget_h × s6_budget_use 안에 끝나는 반복 수로 정한다. milestones는 공식 total_iter 대비 같은 비율로 줄인다.
추론에는 마지막 체크포인트(net_g_latest.pth)를 쓴다."""
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

import gdal_ops as g
from sr_run import TP, WT, official

RE = TP / 'Real-ESRGAN'
OFFICIAL = yaml.safe_load((RE / 'options/finetune_realesrgan_x4plus_pairdata.yml').read_text())
cache = Path(g.CFG['tiles_cache']) / 's6'


def train(name, iters, print_freq):
    """공식 yml을 복사해 반복 횟수·milestones(공식 비율 유지)·로그 간격만 바꿔 학습하고 실험 폴더를 돌려준다."""
    opt = json.loads(json.dumps(OFFICIAL))
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
    total = OFFICIAL['train']['total_iter']
    opt['train']['total_iter'] = iters
    opt['train']['scheduler']['milestones'] = [max(1, round(m * iters / total))
                                               for m in OFFICIAL['train']['scheduler']['milestones']]
    # 로그 간격(학습 설정 아님): 짧은 실행에서도 반복당 시간이 로그에 남게 한다.
    opt['logger']['print_freq'] = print_freq
    yml = cache / f'{name}.yml'
    yml.write_text(yaml.safe_dump(opt, sort_keys=False))
    shutil.copy(yml, g.ROOT / 'results' / f'{name}.yml')
    official('official_sr', ['realesrgan/train.py', '-opt', yml, '--auto_resume'], RE)
    return RE / 'experiments' / name, n_pairs, ds


def sec_per_iter(exp):
    """학습 로그의 반복 로그 시각으로 (마지막 − 첫 로그 시각) / (반복 차)를 계산한다(첫 로그 이전 준비 시간 제외)."""
    pts = []
    for line in max(exp.glob('train_*.log')).read_text().splitlines():
        if ' iter:' in line and 'time (data)' in line:
            t = datetime.strptime(line[:23], '%Y-%m-%d %H:%M:%S,%f')
            pts.append((t, int(line.split(' iter:')[1].split(',')[0])))
    assert len(pts) >= 3, f'too few iteration log lines in {exp}'
    (t0, i0), (t1, i1) = pts[0], pts[-1]
    return (t1 - t0).total_seconds() / (i1 - i0)


def main():
    shutil.rmtree(cache, ignore_errors=True)
    shutil.copytree(g.W / 'tiles/hr/images/train', cache / 'gt')
    shutil.copytree(g.W / 'tiles/lr/images/train', cache / 'lq')
    official('official_sr', ['scripts/generate_meta_info_pairdata.py', '--input', cache / 'gt', cache / 'lq',
                             '--root', cache / 'gt', cache / 'lq', '--meta_info', cache / 'meta_info.txt'], RE)
    name = 'S6_realesrgan_ft_spacenet'
    measure = {}
    if g.CFG['smoke']:
        name += '_smoke'
        iters = g.CFG['s6_smoke_iter']
    else:
        m_iter = g.CFG['s6_measure_iter']
        t0 = time.time()
        exp, _, _ = train(name + '_measure', m_iter, m_iter // 6)
        spi = sec_per_iter(exp)
        iters = int(g.CFG['s6_budget_h'] * 3600 * g.CFG['s6_budget_use'] / spi) // 100 * 100
        measure = {'measure_iter': m_iter, 'sec_per_iter': round(spi, 4), 'measure_seconds': round(time.time() - t0, 1),
                   'budget_h': g.CFG['s6_budget_h'], 'budget_use': g.CFG['s6_budget_use']}
        print('S6 measurement:', measure, '-> total_iter', iters, flush=True)
    t0 = time.time()
    exp, n_pairs, ds = train(name, iters, min(OFFICIAL['logger']['print_freq'], max(iters // 4, 1)))
    sec = time.time() - t0
    net = exp / 'models/net_g_latest.pth'
    # 학습이 실제로 가중치를 바꿨는지 확인한다(바뀌지 않으면 S6 = S4).
    pre = torch.load(WT / 'realesrgan/RealESRGAN_x4plus.pth', map_location='cpu')['params_ema']
    post = torch.load(net, map_location='cpu')['params']
    delta = max((post[k].float() - pre[k].float()).abs().max().item() for k in pre)
    assert delta > 0, f'S6 weights unchanged after {iters} iterations'
    opt = yaml.safe_load((g.ROOT / 'results' / f'{name}.yml').read_text())
    info = {'name': name, 'total_iter': iters, 'milestones': opt['train']['scheduler']['milestones'],
            'official_total_iter': OFFICIAL['train']['total_iter'], **measure, 'n_pairs': n_pairs,
            'dataset_enlarge_ratio': ds['dataset_enlarge_ratio'], 'max_abs_weight_change': delta,
            'seconds': round(sec, 1), 'batch_size_per_gpu': ds['batch_size_per_gpu'], 'gt_size': ds['gt_size'],
            'net_g': str(net), 'train_log': str(max(exp.glob('train_*.log')))}
    (g.W / 's6_finetune.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    shutil.rmtree(cache)


if __name__ == '__main__':
    main()
