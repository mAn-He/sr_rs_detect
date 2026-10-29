"""탐지: D_HR on HR(상한), D_LR on LR(경로 1), D_HR on 각 SR(경로 2) test 타일. Ultralytics 공식 predict API.
타일 픽셀 좌표의 OBB 4점과 conf를 W/preds/<행>.csv에, km²당 탐지 시간을 W/det_timing.csv에 남긴다.
Ultralytics val 지표(타일 단위, test 분할)도 results/det_val_tile.csv에 기록한다(결론은 지리 평가 기준)."""
import time

import geopandas as gpd
import pandas as pd
from ultralytics import YOLO

import gdal_ops as g
from sr_run import ENABLED


def rows():
    run = g.ROOT / 'runs' / g.W.name
    out = [('HR', run / 'D_HR/weights/best.pt', g.W / 'tiles/hr/images/test', g.CFG['imgsz_hr']),
           ('LR', run / 'D_LR/weights/best.pt', g.W / 'tiles/lr/images/test', g.CFG['imgsz_lr'])]
    out += [(m, run / 'D_HR/weights/best.pt', g.W / 'sr' / m / 'images/test', g.CFG['imgsz_hr'])
            for m in g.CFG['sr_methods'] if ENABLED.get(m, True)]
    return out


def main():
    tiles = gpd.read_file(g.W / 'tiles.geojson')
    ids = sorted(tiles[(tiles.split == 'test') & tiles.use].tile_id)
    km2 = len(ids) * (g.CFG['tile_lr'] * g.LR) ** 2 / 1e6
    (g.W / 'preds').mkdir(exist_ok=True)
    timing = []
    for name, weights, img_dir, imgsz in rows():
        files = [str(img_dir / f'{i}.tif') for i in ids]
        model = YOLO(weights)
        # 첫 호출의 CUDA 초기화·모델 준비 시간을 탐지 시간에서 뺀다(타일 1장 예열).
        model.predict(files[:1], imgsz=imgsz, conf=g.CFG['det_conf_ap'], verbose=False)
        t0 = time.time()
        recs = []
        for r in model.predict(files, imgsz=imgsz, conf=g.CFG['det_conf_ap'], stream=True, verbose=False):
            tid = r.path.rsplit('/', 1)[1][:-4]
            xy = r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 8)
            for c, p in zip(r.obb.conf.cpu().numpy(), xy):
                recs.append([tid, float(c), *map(float, p)])
        sec = time.time() - t0
        cols = ['tile_id', 'conf'] + [f'{a}{k}' for k in range(1, 5) for a in ('x', 'y')]
        pd.DataFrame(recs, columns=cols).to_csv(g.W / 'preds' / f'{name}.csv', index=False)
        # 타일 단위 val: 이미지 폴더의 상위에 labels/test가 있어야 하므로 SR 행은 HR 라벨 폴더를 링크한다.
        lab = img_dir.parents[1] / 'labels/test'
        if not lab.exists():
            lab.parent.mkdir(exist_ok=True)
            lab.symlink_to(g.W / ('tiles/lr' if name == 'LR' else 'tiles/hr') / 'labels/test')
        # Ultralytics는 data YAML에 train·val 키를 요구한다. split='test' 평가에서는 test만 읽는다.
        yml = img_dir.parents[1] / 'test.yaml'
        yml.write_text(f'path: {img_dir.parents[1]}\ntrain: images/test\nval: images/test\ntest: images/test\n'
                       'names:\n  0: building\n')
        v = model.val(data=str(yml), split='test', imgsz=imgsz, plots=False, verbose=False,
                      project=str(g.ROOT / 'runs' / g.W.name / 'val'), name=name, exist_ok=True)
        timing.append({'method': name, 'seconds': round(sec, 2), 'km2': km2, 'n_pred': len(recs),
                       'tile_mAP50': float(v.results_dict['metrics/mAP50(B)'])})
        print(timing[-1])
    df = pd.DataFrame(timing)
    df.to_csv(g.W / 'det_timing.csv', index=False)
    df[['method', 'tile_mAP50']].to_csv(g.ROOT / 'results/det_val_tile.csv', index=False)


if __name__ == '__main__':
    main()
