"""탐지(Ultralytics 공식 predict API, conf det_conf_ap, max_det는 label_stats.json).
- val: D_HR on HR, D_LR on LR, D_LR_e<d_lr_sens_epochs> on LR(수렴 민감도) val 타일. 병합 IoU·운영 conf와 민감도 판정에만 쓴다
  (geo_eval.py val).
- test: D_HR on HR(상한), D_LR on LR(경로 1), D_HR on 각 SR(경로 2) test 타일.
타일 픽셀 좌표의 OBB 4점과 conf를 W/preds/<split>/<행>.csv에 남긴다. test에서는 km²당 탐지 시간(모델 로딩·예열 제외)과
로딩·예열 시간을 W/det_timing.csv에, Ultralytics val 지표(타일 단위)를 results/det_val_tile.csv에 기록한다(결론은 지리 평가 기준)."""
import json
import sys
import time

import geopandas as gpd
import pandas as pd
from ultralytics import YOLO

import gdal_ops as g
from sr_run import ENABLED


def rows(split):
    run = g.ROOT / 'runs' / g.W.name
    out = [('HR', run / 'D_HR/weights/best.pt', g.W / f'tiles/hr/images/{split}', g.CFG['imgsz_hr']),
           ('LR', run / 'D_LR/weights/best.pt', g.W / f'tiles/lr/images/{split}', g.CFG['imgsz_lr'])]
    if split == 'val':
        e = g.CFG['d_lr_sens_epochs']
        out.append((f'LR_e{e}', run / f'D_LR_e{e}/weights/best.pt', g.W / 'tiles/lr/images/val', g.CFG['imgsz_lr']))
    if split == 'test':
        out += [(m, run / 'D_HR/weights/best.pt', g.W / 'sr' / m / 'images/test', g.CFG['imgsz_hr'])
                for m in g.CFG['sr_methods'] if ENABLED.get(m, True)]
    return out


def main(split):
    tiles = gpd.read_file(g.W / 'tiles.geojson')
    ids = sorted(tiles[(tiles.split == split) & tiles.use].tile_id)
    km2 = len(ids) * (g.CFG['tile_lr'] * g.LR) ** 2 / 1e6
    max_det = json.loads((g.W / 'label_stats.json').read_text())['max_det']
    kw = {'conf': g.CFG['det_conf_ap'], 'max_det': max_det, 'verbose': False}
    out = g.W / 'preds' / split
    out.mkdir(parents=True, exist_ok=True)
    timing = []
    for name, weights, img_dir, imgsz in rows(split):
        files = [str(img_dir / f'{i}.tif') for i in ids]
        # 모델 로딩과 첫 호출의 CUDA 초기화·모델 준비(타일 1장 예열)는 탐지 시간에서 빼고 따로 기록한다.
        t0 = time.time()
        model = YOLO(weights)
        model.predict(files[:1], imgsz=imgsz, **kw)
        load = time.time() - t0
        t0 = time.time()
        recs = []
        for r in model.predict(files, imgsz=imgsz, stream=True, **kw):
            tid = r.path.rsplit('/', 1)[1][:-4]
            xy = r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 8)
            for c, p in zip(r.obb.conf.cpu().numpy(), xy):
                recs.append([tid, float(c), *map(float, p)])
        sec = time.time() - t0
        cols = ['tile_id', 'conf'] + [f'{a}{k}' for k in range(1, 5) for a in ('x', 'y')]
        df = pd.DataFrame(recs, columns=cols)
        df.to_csv(out / f'{name}.csv', index=False)
        per_tile = df.groupby('tile_id').size()
        print(f'{split} {name}: {len(df)} predictions, max per tile {per_tile.max()}, '
              f'{(per_tile >= max_det).sum()} tiles at max_det {max_det}')
        if split == 'val':
            continue
        # 타일 단위 val: 이미지 폴더의 상위에 labels/test가 있어야 하므로 SR 행은 HR 라벨 폴더를 링크한다.
        lab = img_dir.parents[1] / 'labels/test'
        if not lab.exists():
            lab.parent.mkdir(exist_ok=True)
            lab.symlink_to(g.W / ('tiles/lr' if name == 'LR' else 'tiles/hr') / 'labels/test')
        # Ultralytics는 data YAML에 train·val 키를 요구한다. split='test' 평가에서는 test만 읽는다.
        yml = img_dir.parents[1] / 'test.yaml'
        yml.write_text(f'path: {img_dir.parents[1]}\ntrain: images/test\nval: images/test\ntest: images/test\n'
                       'names:\n  0: building\n')
        v = model.val(data=str(yml), split='test', imgsz=imgsz, max_det=max_det, plots=False, verbose=False,
                      project=str(g.ROOT / 'runs' / g.W.name / 'val'), name=name, exist_ok=True)
        timing.append({'method': name, 'seconds': round(sec, 2), 'load_sec': round(load, 2), 'km2': km2,
                       'n_pred': len(df), 'max_pred_per_tile': int(per_tile.max()),
                       'tiles_at_max_det': int((per_tile >= max_det).sum()),
                       'tile_mAP50': float(v.results_dict['metrics/mAP50(B)'])})
        print(timing[-1])
    if split == 'test':
        df = pd.DataFrame(timing)
        df.to_csv(g.W / 'det_timing.csv', index=False)
        df[['method', 'tile_mAP50']].to_csv(g.ROOT / 'results/det_val_tile.csv', index=False)


if __name__ == '__main__':
    main(sys.argv[1])
