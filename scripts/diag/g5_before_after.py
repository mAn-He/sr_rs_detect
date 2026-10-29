"""MS 동서 메타데이터 정정 전/후 비교 그림(칩 단위, PS-RGB 0.27e-5° 격자).

홀수 열 칩 3장(정정 대상), 짝수 열 칩 2장(대조군)을 phase.csv의 정정 전 중심 dx가 그룹 중앙값에 가깝고
건물이 15채 이상인 칩에서 고른다. 각 칩: [정정 전 | 정정 후] 패널. LR(MS 5,3,2)을 HR 격자로 최근접 확대한 영상 위에
HR Canny(초록), LR Canny(빨강, LR을 HR 격자로 bilinear 확대 후), 건물 라벨 윤곽(노랑). 건물 근처 100×100 HR px를 6배 최근접 확대.
사용: python scripts/diag/g5_before_after.py <phase.csv> <out_dir>
"""
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from osgeo import gdal

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g5_investigate import RAW, ms_path, ps_path
from g5_phase import PS, measure

MSB = [5, 3, 2]
CROP, ZOOM = 100, 6


def stretch(img):
    lo, hi = np.percentile(img, (2, 98), axis=(1, 2), keepdims=True)
    return np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def canny(lum):
    m = float(np.median(lum))
    return cv2.Canny(lum, int(0.66 * m), int(1.33 * m)) > 0


def ms_on_ps(cid, x0, alg):
    """MS 칩(동서 원점 x0, 동서 화소 4×PS 또는 메타데이터)을 PS 칩 격자로 리샘플."""
    ms, ps = gdal.Open(ms_path(cid)), gdal.Open(ps_path(cid))
    g, gp = list(ms.GetGeoTransform()), ps.GetGeoTransform()
    mem = gdal.Translate('', ms, format='MEM', bandList=MSB)
    if x0 is not None:
        g[0], g[1] = x0, 4 * PS
        mem.SetGeoTransform(g)
    n = ps.RasterXSize
    b = (gp[0], gp[3] + gp[5] * n, gp[0] + gp[1] * n, gp[3])
    return gdal.Warp('', mem, format='MEM', outputBounds=b, width=n, height=n, resampleAlg=alg).ReadAsArray().astype(np.float64)


def panel(ax, hr8, lr_near, lr_bil, polys, win, title):
    r0, c0 = win
    sl = np.s_[r0:r0 + CROP, c0:c0 + CROP]
    img = stretch(lr_near)[:, sl[0], sl[1]].transpose(1, 2, 0)
    img = cv2.resize(img, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST).copy()
    eh = canny(cv2.cvtColor(hr8.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY))[sl]
    el = canny(cv2.cvtColor(stretch(lr_bil).transpose(1, 2, 0), cv2.COLOR_RGB2GRAY))[sl]
    for e, col in ((eh, (0, 255, 0)), (el, (255, 0, 0))):
        img[cv2.resize(e.astype(np.uint8), None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST) > 0] = col
    for p in polys:
        pts = ((p - [c0, r0]) * ZOOM + ZOOM / 2).round().astype(np.int32)
        cv2.polylines(img, [pts], True, (255, 255, 0), 2)
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def main():
    ph = pd.read_csv(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    nb = {c: len(gpd.read_file(RAW / f'geojson_buildings/SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img{c}.geojson'))
          for c in ph.chip_id}
    ph['n_bld'] = ph.chip_id.map(nb)
    picks = []
    for par, k in ((1, 3), (0, 2)):
        g = ph[(ph.col % 2 == par) & (ph.n_bld >= 15)]
        picks += g.assign(d=(g.orig_tx - g.orig_tx.median()).abs()).sort_values('d').chip_id.head(k).tolist()
    for cid in picks:
        r = ph[ph.chip_id == cid].iloc[0]
        ps = gdal.Open(ps_path(cid))
        gp = ps.GetGeoTransform()
        hr8 = stretch(ps.ReadAsArray().astype(np.float64))
        bld = gpd.read_file(RAW / f'geojson_buildings/SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img{cid}.geojson')
        polys = [np.c_[(np.array(p.exterior.xy[0]) - gp[0]) / gp[1], (np.array(p.exterior.xy[1]) - gp[3]) / gp[5]]
                 for g in bld.geometry if g is not None and not g.is_empty for p in getattr(g, 'geoms', [g])]
        cen = min(polys, key=lambda p: np.hypot(*(p.mean(0) - 325)))
        cc, rr = np.clip(cen.mean(0).round().astype(int) - CROP // 2, 0, 650 - CROP)
        x0 = gp[0] - 2 * PS * (r.col % 2)
        before, after = measure(cid), measure(cid, px_x=4 * PS, shift_x_ps=2 * (r.col % 2))
        fig, ax = plt.subplots(1, 2, figsize=(12, 6.6), constrained_layout=True)
        par = 'odd' if r.col % 2 else 'even'
        for a, tag, xo, m in ((ax[0], 'before (metadata)', None, before), (ax[1], 'after (x rule)', x0, after)):
            panel(a, hr8, ms_on_ps(cid, xo, 'near'), ms_on_ps(cid, xo, 'bilinear'), polys, (rr, cc),
                  f'img{cid}  col {int(r.col)} ({par})  {tag}\ndx {m["tx"]:+.3f}  dy {m["ty"]:+.3f} MS px (chip centre, ECC)')
        fig.suptitle('LR (MS 5,3,2, nearest to HR grid) + HR Canny (green) + LR Canny (red) + building labels (yellow), '
                     f'{CROP}x{CROP} HR px x{ZOOM}', fontsize=9)
        fig.savefig(out / f'img{cid}_col{par}.png', dpi=100)
        plt.close(fig)
        print('saved', out / f'img{cid}_col{par}.png', par, before['tx'], after['tx'])
    (out / 'caption.txt').write_text(
        'MS 동서 메타데이터 정정 전/후: 영상 픽셀값은 바뀌지 않고 geotransform(동서 화소 4×PS, 홀수 열 원점 −2 PS px)만 바뀐다. '
        '초록 HR Canny, 빨강 LR Canny(LR bilinear 확대 후), 노랑 건물 라벨, 배경은 LR 최근접 확대.\n')


if __name__ == '__main__':
    main()
