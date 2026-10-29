"""D_HR(hr) / D_LR(lr) 학습: Ultralytics 공식 OBB API. 학습 직전 해당 타일을 tiles_cache로 복사하고 끝나면 삭제한다."""
import json
import shutil
import sys
from pathlib import Path

import psutil
import torch
from ultralytics import YOLO

import gdal_ops as g

KIND = sys.argv[1]
assert KIND in ('hr', 'lr')
NAME = f'D_{KIND.upper()}'
cache = Path(g.CFG['tiles_cache']) / KIND
shutil.rmtree(cache, ignore_errors=True)
for sub in ('images', 'labels'):
    for split in ('train', 'val'):
        shutil.copytree(g.W / 'tiles' / KIND / sub / split, cache / sub / split)
data = cache / 'data.yaml'
data.write_text(f'path: {cache}\ntrain: images/train\nval: images/val\nnames:\n  0: building\n')
imgsz = g.CFG[f'imgsz_{KIND}']
# 학습 데이터(디코딩 후 uint8)가 사용 가능 RAM의 절반 안에 들어가면 Ultralytics cache='ram'을 쓴다.
n_img = sum(1 for _ in (cache / 'images/train').iterdir()) + sum(1 for _ in (cache / 'images/val').iterdir())
need_gb = n_img * imgsz * imgsz * 3 / 1e9
cache_opt = 'ram' if need_gb < psutil.virtual_memory().available / 2e9 else False
print(f'{NAME}: {n_img} images, {need_gb:.2f} GB decoded, cache={cache_opt}')
model = YOLO(Path(g.CFG['weights_dir']) / 'yolo' / g.CFG['yolo_weights'])
model.train(data=str(data), imgsz=imgsz, epochs=g.CFG['smoke_epochs'] if g.CFG['smoke'] else g.CFG['epochs'],
            patience=g.CFG['patience'], batch=g.CFG['batch'], workers=g.CFG['workers'], seed=g.CFG['seed'],
            cache=cache_opt, project=str(g.ROOT / 'runs' / g.W.name), name=NAME, exist_ok=True)
tr = model.trainer
run_dir = Path(tr.save_dir)
out = g.ROOT / 'results'
shutil.copy(run_dir / 'results.csv', out / f'det_{NAME}_results.csv')
shutil.copy(run_dir / 'results.png', g.FIG / f'det_{NAME}_results.png')
info = {'name': NAME, 'imgsz': imgsz, 'batch_actual': int(tr.batch_size), 'epochs_run': int(tr.epoch) + 1,
        'best': str(tr.best),
        'n_train_images': len(list((cache / 'images/train').iterdir())), 'cache': cache_opt,
        'val_metrics': {k: float(v) for k, v in tr.metrics.items()}, 'gpu': torch.cuda.get_device_name(0)}
(out / f'det_{NAME}.json').write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
shutil.rmtree(cache)
