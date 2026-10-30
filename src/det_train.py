"""D_HR(hr) / D_LR(lr) 학습: Ultralytics 공식 OBB API. lr_sens는 D_LR 수렴 민감도용으로 epochs만 d_lr_sens_epochs로 바꿔
D_LR_e<epochs>라는 별도 이름으로 학습한다(D_LR 가중치를 덮어쓰지 않는다). 학습 직전 해당 타일을 tiles_cache로 복사하고 끝나면 삭제한다
(cache='disk'의 .npy도 tiles_cache 안에 생겨 함께 삭제된다)."""
import json
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

import gdal_ops as g

assert sys.argv[1] in ('hr', 'lr', 'lr_sens')
KIND = sys.argv[1].removesuffix('_sens')
EPOCHS = g.CFG['smoke_epochs'] if g.CFG['smoke'] else g.CFG['epochs']
NAME = f'D_{KIND.upper()}'
if sys.argv[1] == 'lr_sens':
    EPOCHS = g.CFG['d_lr_sens_epochs']
    NAME = f'D_LR_e{EPOCHS}'
cache = Path(g.CFG['tiles_cache']) / KIND
shutil.rmtree(cache, ignore_errors=True)
for sub in ('images', 'labels'):
    for split in ('train', 'val'):
        shutil.copytree(g.W / 'tiles' / KIND / sub / split, cache / sub / split)
data = cache / 'data.yaml'
data.write_text(f'path: {cache}\ntrain: images/train\nval: images/val\nnames:\n  0: building\n')
imgsz = g.CFG[f'imgsz_{KIND}']
# 학습 val의 max_det: labels.py가 train·val 타일당 GT 최대 개수로 정한 값(추론과 같다).
max_det = json.loads((g.W / 'label_stats.json').read_text())['max_det']
model = YOLO(Path(g.CFG['weights_dir']) / 'yolo' / g.CFG['yolo_weights'])
model.train(data=str(data), imgsz=imgsz, epochs=EPOCHS,
            patience=g.CFG['patience'], batch=g.CFG['batch'], workers=g.CFG['workers'], seed=g.CFG['seed'],
            deterministic=g.CFG['deterministic'], cache=g.CFG['det_cache'], max_det=max_det,
            project=str(g.ROOT / 'runs' / g.W.name), name=NAME, exist_ok=True)
tr = model.trainer
run_dir = Path(tr.save_dir)
out = g.ROOT / 'results'
shutil.copy(run_dir / 'results.csv', out / f'det_{NAME}_results.csv')
shutil.copy(run_dir / 'results.png', g.FIG / f'det_{NAME}_results.png')
info = {'name': NAME, 'imgsz': imgsz, 'batch_actual': int(tr.batch_size), 'epochs_run': int(tr.epoch) + 1,
        'best': str(tr.best),
        'n_train_images': len(list((cache / 'images/train').glob('*.tif'))),
        'args': {k: getattr(tr.args, k) for k in ('seed', 'deterministic', 'cache', 'max_det', 'workers', 'epochs',
                                                    'patience')},
        'val_metrics': {k: float(v) for k, v in tr.metrics.items()}, 'gpu': torch.cuda.get_device_name(0)}
(out / f'det_{NAME}.json').write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
shutil.rmtree(cache)
