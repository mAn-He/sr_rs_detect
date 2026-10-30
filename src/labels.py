"""G10: 건물 GeoJSON(UTM) → 타일 창 clip(원래 면적의 label_min_frac 미만 제외) → 픽셀 좌표(~transform)
→ cv2.minAreaRect/boxPoints → 0~1 정규화 → YOLO OBB txt. HR(1024)·LR(256) 라벨, 건물별 면적·partialBuilding CSV."""
import json

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

import gdal_ops as g


def obb_label(geom, transform, size):
    """UTM 폴리곤 → 타일 픽셀 좌표의 최소 회전 사각형 4점, 0~1 정규화한 (4, 2)."""
    parts = [p for p in getattr(geom, 'geoms', [geom]) if p.geom_type == 'Polygon']
    xy = np.vstack([np.asarray(p.exterior.coords)[:, :2] for p in parts])
    col, row = ~transform @ (xy[:, 0], xy[:, 1])
    pts = cv2.boxPoints(cv2.minAreaRect(np.c_[col, row].astype(np.float32)))
    return np.clip(pts / size, 0, 1)


def main():
    tiles = gpd.read_file(g.W / 'tiles.geojson')
    tiles = tiles[tiles.use]
    bld = gpd.read_file(g.W / 'buildings_utm.geojson').reset_index(names='building_id')
    hr_px = 4 * g.CFG['tile_lr']
    recs = []
    for t in tiles.itertuples():
        with rasterio.open(g.W / f'tiles/hr/images/{t.split}/{t.tile_id}.tif') as src:
            tr = src.transform
        cand = bld.iloc[bld.sindex.query(t.geometry, predicate='intersects')]
        lines = []
        for b in cand.itertuples():
            clip = b.geometry.intersection(t.geometry)
            frac = clip.area / b.geometry.area
            if frac < g.CFG['label_min_frac']:
                continue
            pts = obb_label(clip, tr, hr_px)
            lines.append('0 ' + ' '.join(f'{v:.6f}' for v in pts.ravel()))
            recs.append({'tile_id': t.tile_id, 'split': t.split, 'block_id': t.block_id, 'building_id': b.building_id,
                         'area_m2': b.area_m2, 'clip_frac': round(frac, 4), 'partialBuilding': b.partialBuilding})
        # 같은 지리 창이라 정규화 좌표는 HR·LR이 같다.
        for kind in ('hr', 'lr'):
            p = g.W / f'tiles/{kind}/labels/{t.split}/{t.tile_id}.txt'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(''.join(x + '\n' for x in lines))
    df = pd.DataFrame(recs)
    df.to_csv(g.W / 'labels_buildings.csv', index=False)
    stats(df, bld, tiles)
    overlays(tiles, df)


def stats(df, bld, tiles):
    """건물 면적 분포, HR·LR 건물당 픽셀 수 분포, 분할별 건물 수(라벨 인스턴스·고유 건물),
    타일당 GT 최대 개수와 max_det(train·val 타일 기준, test 타일이 넘지 않는지 확인)."""
    u = df.drop_duplicates('building_id')
    per_tile = df.groupby('tile_id').size().reindex(tiles.tile_id, fill_value=0)
    split_of = tiles.set_index('tile_id').split
    max_gt = int(per_tile[split_of != 'test'].max())
    max_det = max(g.CFG['max_det_min'], g.CFG['max_det_factor'] * max_gt)
    assert per_tile.max() <= max_det, (per_tile.max(), max_det)
    edges = [0, *g.CFG['area_bins'], np.inf]
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    out = {
        'area_m2_quantiles': dict(zip(map(str, q), u.area_m2.quantile(q).round(1))),
        'area_bins_count': pd.cut(u.area_m2, edges).value_counts().sort_index().rename(str).to_dict(),
        'hr_px_per_building_quantiles': dict(zip(map(str, q), (u.area_m2 / g.HR**2).quantile(q).round(0))),
        'lr_px_per_building_quantiles': dict(zip(map(str, q), (u.area_m2 / g.LR**2).quantile(q).round(1))),
        'per_split': {s: {'tiles': int((tiles.split == s).sum()), 'label_instances': int((df.split == s).sum()),
                          'unique_buildings': int(u[u.split == s].shape[0])} for s in g.SPLITS},
        'buildings_in_used_blocks_total': len(u), 'buildings_all': len(bld),
        'max_gt_per_tile': {s: int(per_tile[split_of == s].max()) for s in g.SPLITS}, 'max_det': max_det,
    }
    (g.W / 'label_stats.json').write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


def overlays(tiles, df):
    """라벨 오버레이 PNG: 건물 수가 많은 타일 5장, HR(1024)·LR(256, 최근접 4배) 각각."""
    top = df.groupby('tile_id').size().sort_values(ascending=False).index[:5]
    for tid in top:
        t = tiles[tiles.tile_id == tid].iloc[0]
        for kind in ('hr', 'lr'):
            with rasterio.open(g.W / f'tiles/{kind}/images/{t.split}/{tid}.tif') as src:
                img = np.ascontiguousarray(src.read().transpose(1, 2, 0)[:, :, ::-1])
            if kind == 'lr':
                img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            n = img.shape[0]
            lab = np.loadtxt(g.W / f'tiles/{kind}/labels/{t.split}/{tid}.txt', ndmin=2)[:, 1:].reshape(-1, 4, 2)
            cv2.polylines(img, list(np.rint(lab * n).astype(np.int32)), True, (0, 255, 255), 2)
            png = g.FIG / f'g10_labels_{kind}_{tid}.png'
            cv2.imwrite(str(png), img)
            print('PNG', png)


if __name__ == '__main__':
    main()
