"""G11: 타일 예측 → 전역 좌표(타일 transform, UTM) → 블록별 cv2.dnn.NMSBoxesRotated(타일은 블록 안에서만 겹친다)
→ UTM 회전 사각형 폴리곤 → EPSG:4326 GeoJSON.
- val: 후보 IoU(nms_iou_candidates)마다 W/preds/val/<행>_iou<IoU>.geojson(geo_eval.py val이 IoU를 고른다).
- test: results/det_calibration.json의 IoU 하나로 W/preds/test/<행>.geojson."""
import json
import sys

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Polygon

import gdal_ops as g
from det_infer import rows


def merge(p, img_dir, tiles, te, iou):
    xs, ys = p[[f'x{k}' for k in range(1, 5)]].to_numpy(), p[[f'y{k}' for k in range(1, 5)]].to_numpy()
    ux, uy = np.empty_like(xs), np.empty_like(ys)
    for tid, idx in p.groupby('tile_id').indices.items():
        with rasterio.open(img_dir / f'{tid}.tif') as src:
            ux[idx], uy[idx] = src.transform @ (xs[idx], ys[idx])
    keep_all = []
    for idx in p.groupby(p.tile_id.map(tiles.block_id)).indices.values():
        # 공통 범위 좌상단 기준 m 단위 평면 좌표(y 아래 방향)에서 회전 NMS
        rects = [cv2.minAreaRect(np.c_[ux[i] - te[0], te[3] - uy[i]].astype(np.float32)) for i in idx]
        keep = cv2.dnn.NMSBoxesRotated(rects, p.conf.to_numpy()[idx].tolist(), g.CFG['det_conf_ap'], iou)
        keep_all += [idx[k] for k in np.asarray(keep, int).ravel()]
    keep_all = np.sort(np.array(keep_all, dtype=int))
    geoms = [Polygon(zip(ux[i], uy[i])) for i in keep_all]
    return gpd.GeoDataFrame({'conf': p.conf.to_numpy()[keep_all], 'tile_id': p.tile_id.to_numpy()[keep_all]},
                            geometry=geoms, crs=g.UTM).to_crs('EPSG:4326')


def main(split):
    tiles = gpd.read_file(g.W / 'tiles.geojson').set_index('tile_id')
    te = json.loads((g.W / 'extent.json').read_text())['te']
    d = g.W / 'preds' / split
    if split == 'val':
        jobs = [(iou, f'_iou{iou}') for iou in g.CFG['nms_iou_candidates']]
    else:
        jobs = [(json.loads((g.ROOT / 'results/det_calibration.json').read_text())['nms_iou'], '')]
    for name, _, img_dir, _ in rows(split):
        p = pd.read_csv(d / f'{name}.csv')
        for iou, suffix in jobs:
            out = merge(p, img_dir, tiles, te, iou)
            out.to_file(d / f'{name}{suffix}.geojson', driver='GeoJSON')
            print(f'G11 {split} {name} IoU {iou}: {len(p)} tile predictions -> {len(out)} after rotated NMS')


if __name__ == '__main__':
    main(sys.argv[1])
