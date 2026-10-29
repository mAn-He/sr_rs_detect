"""G5 측정기 수정 전 확인용: G5와 같은 창을 짝수 최적 DFT 크기로 잘라 phaseCorrelate(홀짝 편향 제거)와 ECC로 다시 잰다.

사용: python scripts/diag/g5_fixed.py <out_dir> → <out_dir>/blocks_fixed.csv (현재 data/<mode>/ VRT 기준)
"""
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g5_investigate import ecc

import gdal_ops as g


def even_opt(n):
    m = n
    while m % 2 or cv2.getOptimalDFTSize(m) != m:
        m -= 1
    return m


def block(args):
    geom, wins = args
    x0, _, _, y1 = geom.bounds
    with rasterio.open(g.W / 'hr_utm.vrt') as h, rasterio.open(g.W / 'lr_utm.vrt') as lr:
        hr = h.read(window=g.window(h, geom)).astype(np.float64)
        lo = lr.read(window=g.window(lr, geom)).astype(np.float64)
    n = lo.shape[1]
    a = hr.reshape(3, n, 4, n, 4).mean(axis=(2, 4)).mean(axis=0)
    b = lo.mean(axis=0)
    res = []
    for wx0, wy0, wx1, wy1 in wins:
        c0, c1 = math.ceil((wx0 - x0) / 1.2), math.floor((wx1 - x0) / 1.2)
        r0, r1 = math.ceil((y1 - wy1) / 1.2), math.floor((y1 - wy0) / 1.2)
        if min(c1 - c0, r1 - r0) < 64:
            continue
        w, hh = even_opt(c1 - c0), even_opt(r1 - r0)
        ua, ub = np.ascontiguousarray(a[r0:r0 + hh, c0:c0 + w]), np.ascontiguousarray(b[r0:r0 + hh, c0:c0 + w])
        (dx, dy), _ = cv2.phaseCorrelate(ua, ub, cv2.createHanningWindow((w, hh), cv2.CV_64F))
        ex, ey = ecc(ua, ub)
        res.append((dx, dy, ex, ey))
    return res


if __name__ == '__main__':
    blocks = g.used_blocks()
    chips = gpd.read_file(g.W / 'chips.geojson')
    jobs = []
    for bg in blocks.geometry:
        inner = [c.buffer(-6).intersection(bg) for c in chips.geometry[chips.intersects(bg)]]
        jobs.append((bg, [x.bounds for x in inner if not x.is_empty]))
    with ProcessPoolExecutor(12) as ex:
        out = list(ex.map(block, jobs))
    rows = []
    for (_, b), r in zip(blocks.iterrows(), out):
        r = np.array(r)
        rows.append({'block_id': b.block_id, 'split': b.split, 'n_windows': len(r), 'pc_dx': np.median(r[:, 0]), 'pc_dy': np.median(r[:, 1]),
                     'ecc_dx': np.median(r[:, 2]), 'ecc_dy': np.median(r[:, 3])})
    d = pd.DataFrame(rows)
    d['pc_shift'] = np.hypot(d.pc_dx, d.pc_dy)
    d['ecc_shift'] = np.hypot(d.ecc_dx, d.ecc_dy)
    d.to_csv(f'{sys.argv[1]}/blocks_fixed.csv', index=False)
    old = pd.read_csv(g.W / 'g5_registration.csv')
    q = lambda s: f'median {s.median():+.3f} IQR [{s.quantile(.25):+.3f}, {s.quantile(.75):+.3f}] max|.| {s.abs().max():.3f}'
    print('OLD G5 (biased) dx', q(old.dx), '| dy', q(old.dy), '| shift', q(old['shift']), '| >0.5:', (old['shift'] > 0.5).sum())
    for m in ('pc', 'ecc'):
        print(f'FIXED {m:3s} dx', q(d[m + '_dx']), '| dy', q(d[m + '_dy']), '| shift', q(d[m + '_shift']), '| >0.5:', (d[m + '_shift'] > 0.5).sum(), '/', len(d))
    print('by split (ecc >0.5):', d.groupby('split').apply(lambda t: f"{(t.ecc_shift > 0.5).sum()}/{len(t)}").to_dict())
