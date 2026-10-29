"""G9: 576 m 블록 안 307.2 m 지리 창 타일링(블록당 2×2, LR 시작점 0·224). HR·LR 타일은 모든 분할, SR 타일(S0·S1)은 test만.
train·val 타일은 유효 비율 min_valid_frac 이상만 저장하고(M7), 무효 픽셀은 제품별 채움값(M2)."""
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

import gdal_ops as g

T = g.CFG['tile_lr'] * g.LR  # 307.2 m


def tile_windows(blocks, fp):
    rows = []
    starts = g.CFG['tile_starts_lr']
    assert starts[-1] + g.CFG['tile_lr'] == round(g.B / g.LR), 'last tile must end at the block edge'
    for blk in blocks.itertuples():
        bx0, _, _, by1 = blk.geometry.bounds
        for i, sr in enumerate(starts):
            for j, sc in enumerate(starts):
                x0, y1 = bx0 + sc * g.LR, by1 - sr * g.LR
                assert np.allclose([x0 / g.LR, y1 / g.LR], np.rint([x0 / g.LR, y1 / g.LR]), atol=1e-6), (x0, y1)
                geom = box(x0, y1 - T, x0 + T, y1)
                valid = geom.intersection(fp).area / geom.area
                rows.append({'tile_id': f'b{blk.block_id:03d}_r{i}c{j}', 'block_id': blk.block_id, 'split': blk.split,
                             'valid_frac': round(valid, 4),
                             'use': blk.split == 'test' or valid >= g.CFG['min_valid_frac'], 'geometry': geom})
    return gpd.GeoDataFrame(rows, crs=g.UTM)


def cut(args):
    tid, split, geom, fp, hr_fill, lr_fill = args
    for kind, src, fill in (('hr', g.W / 'hr_8bit.vrt', hr_fill), ('lr', g.W / 'lr_8bit.vrt', lr_fill)):
        img, mask = g.block_read(src, geom, fp)
        with rasterio.open(src) as s:
            t = s.window_transform(g.window(s, geom))
        g.write_tif(g.W / 'tiles' / kind / 'images' / split / f'{tid}.tif', g.filled(img, mask, fill), t)
    if split == 'test':
        blk = int(tid[1:4])
        for m in ('S0_gdal_cubic', 'S1_gdal_lanczos'):
            with rasterio.open(g.W / 'sr_blocks' / m / f'block_{blk:03d}.tif') as s:
                win = g.window(s, geom)
                img, t = s.read(window=win), s.window_transform(win)
            g.write_tif(g.W / 'sr' / m / 'images/test' / f'{tid}.tif', img, t)
    return tid


def main():
    blocks = g.used_blocks()
    fp = gpd.read_file(g.W / 'footprint.geojson').geometry.iloc[0]
    tiles = tile_windows(blocks, fp)
    tiles.to_file(g.W / 'tiles.geojson', driver='GeoJSON')
    hr_fill, lr_fill = g.fills()
    jobs = [(t.tile_id, t.split, t.geometry, fp, hr_fill, lr_fill) for t in tiles[tiles.use].itertuples()]
    with ProcessPoolExecutor(g.CFG['n_proc']) as ex:
        done = list(ex.map(cut, jobs))
    print(f'G9: {len(done)} tiles written')
    q = tiles.groupby('split').valid_frac.describe(percentiles=[0.1, 0.25, 0.5, 0.75]).reindex(g.SPLITS)
    q['used'] = tiles.groupby('split').use.sum()
    print(q.round(3).to_string())
    q.to_csv(g.W / 'tile_summary.csv')
    # 저장된 HR 타일과 LR 타일이 같은 지리 창이고 LR × 4 = HR인지 확인한다.
    for t in tiles[tiles.use].itertuples():
        with rasterio.open(g.W / f'tiles/hr/images/{t.split}/{t.tile_id}.tif') as h, \
                rasterio.open(g.W / f'tiles/lr/images/{t.split}/{t.tile_id}.tif') as lr:
            assert h.shape == (lr.height * 4, lr.width * 4) == (4 * g.CFG['tile_lr'],) * 2
            assert h.transform.almost_equals(lr.transform * rasterio.Affine.scale(1 / 4), 1e-9)


if __name__ == '__main__':
    main()
