"""G5 원인 조사: 칩별 HR↔LR 오프셋(원본 MS 격자·UTM, phaseCorrelate·ECC)과 합성 이동 복원 테스트.

사용: python scripts/diag/g5_investigate.py <out_dir>  → <out_dir>/chips.csv, synth.csv (docs/decisions.md G5 참조)
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
from osgeo import gdal, osr
from rasterio.windows import Window

gdal.UseExceptions()
REPO = Path(__file__).resolve().parents[2]
W = REPO / 'data/full'
RAW = REPO / 'data/raw'
OUT = Path(sys.argv[1])
LR, HR = 1.2, 0.3
MSB = [5, 3, 2]
CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)


def norm(x):
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-9)


def pc(a, b):
    """phaseCorrelate(a, b) with Hanning window -> (dx, dy)."""
    han = cv2.createHanningWindow(a.shape[::-1], cv2.CV_64F)
    (dx, dy), _ = cv2.phaseCorrelate(np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64), han)
    return dx, dy


def ecc(a, b):
    """findTransformECC(template=a, input=b), translation -> (tx, ty)."""
    w = np.eye(2, 3, dtype=np.float32)
    _, w = cv2.findTransformECC(norm(a), norm(b), w, cv2.MOTION_TRANSLATION, CRIT, None, 5)
    return float(w[0, 2]), float(w[1, 2])


def ms_path(c):
    return str(RAW / f'MS/SN2_buildings_train_AOI_2_Vegas_MS_img{c}.tif')


def ps_path(c):
    return str(RAW / f'PS-RGB/SN2_buildings_train_AOI_2_Vegas_PS-RGB_img{c}.tif')


def native(c, px_x=None, px_y=None, shift_x_ps=0.0, shift_y_ps=0.0):
    """PS를 MS 격자로 average 축소한 밝기와 MS 밝기. MS geotransform 가설: 화소 크기 px_x/px_y(°),
    원점을 서쪽으로 shift_x_ps, 북쪽으로 shift_y_ps PS 화소만큼 옮김."""
    m = gdal.Open(ms_path(c))
    g = list(m.GetGeoTransform())
    g[0] -= shift_x_ps * 2.7e-6
    g[3] += shift_y_ps * 2.7e-6
    if px_x is not None:
        g[1] = px_x
    if px_y is not None:
        g[5] = -px_y
    n = m.RasterXSize
    bounds = (g[0], g[3] + g[5] * n, g[0] + g[1] * n, g[3])
    p = gdal.Warp('', ps_path(c), format='MEM', outputBounds=bounds, width=n, height=n, resampleAlg='average')
    a = p.ReadAsArray().astype(np.float64).mean(0)
    b = m.ReadAsArray()[[i - 1 for i in MSB]].astype(np.float64).mean(0)
    return a, b


def chip_job(args):
    cid, x0, y0, x1, y1, lrt = args
    r = {'chip_id': cid}
    # 원본 MS 격자(워프 없음), 가장자리 4 px 제외
    a, b = native(cid)
    k = 4
    a, b = a[k:-k, k:-k], b[k:-k, k:-k]
    r['nat_pc_dx'], r['nat_pc_dy'] = pc(a, b)
    r['nat_ecc_dx'], r['nat_ecc_dy'] = ecc(a, b)
    h = a.shape[1] // 2
    r['nat_left_dx'], _ = pc(a[:, :h], b[:, :h])
    r['nat_right_dx'], _ = pc(a[:, h:], b[:, h:])
    _, r['nat_top_dy'] = pc(a[:h], b[:h])
    _, r['nat_bottom_dy'] = pc(a[h:], b[h:])
    # 가설: MS 화소 = 4 × PS(1.08e-5°), 원점 같음 / 원점을 1 PS px 앞으로(가운데 정렬)
    for name, sh in (('h4_origin', 0.0), ('h4_center', 1.0)):
        a2, b2 = native(cid, 4 * 2.7e-6, 4 * 2.7e-6, sh, sh)
        a2, b2 = a2[k:-k, k:-k], b2[k:-k, k:-k]
        r[f'{name}_dx'], r[f'{name}_dy'] = pc(a2, b2)
    # UTM 파이프라인(G5와 같은 입력): 칩 안쪽 6 m 창, HR 4x4 평균 vs LR(cubic VRT)
    c0, r0 = math.ceil((x0 - lrt[0]) / LR), math.ceil((lrt[1] - y1) / LR)
    c1, r1 = math.floor((x1 - lrt[0]) / LR), math.floor((lrt[1] - y0) / LR)
    with rasterio.open(W / 'hr_utm.vrt') as hs, rasterio.open(W / 'lr_utm.vrt') as ls:
        lo = ls.read(window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float64).mean(0)
        hr = hs.read(window=Window(4 * c0, 4 * r0, 4 * (c1 - c0), 4 * (r1 - r0))).astype(np.float64).mean(0)
    hd = hr.reshape(lo.shape[0], 4, lo.shape[1], 4).mean(axis=(1, 3))
    r['utm_pc_dx'], r['utm_pc_dy'] = pc(hd, lo)
    r['utm_ecc_dx'], r['utm_ecc_dy'] = ecc(hd, lo)
    return r


def synth_job(args):
    """HR(0.3 m UTM) 창 → gdalwarp -r average로 1.2 m(기준) / 격자를 s LR px 옮긴 1.2 m. 내용 이동 (+s 열, +s 행)."""
    cid, x0, y0, x1, y1, lrt = args
    c0, r0 = math.ceil((x0 - lrt[0]) / LR), math.ceil((lrt[1] - y1) / LR)
    c1, r1 = math.floor((x1 - lrt[0]) / LR), math.floor((lrt[1] - y0) / LR)
    bx0, by1 = lrt[0] + c0 * LR, lrt[1] - r0 * LR
    bx1, by0 = lrt[0] + c1 * LR, lrt[1] - r1 * LR
    # HR 창에 여유 4 LR px를 두고 MEM으로
    pad = 4
    with rasterio.open(W / 'hr_utm.vrt') as hs:
        hr = hs.read(window=Window(4 * (c0 - pad), 4 * (r0 - pad), 4 * (c1 - c0 + 2 * pad), 4 * (r1 - r0 + 2 * pad))).astype(np.float32)
    mem = gdal.GetDriverByName('MEM').Create('', hr.shape[2], hr.shape[1], 3, gdal.GDT_Float32)
    mem.SetGeoTransform((bx0 - pad * LR, HR, 0, by1 + pad * LR, 0, -HR))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32611)
    mem.SetProjection(srs.ExportToWkt())
    for i in range(3):
        mem.GetRasterBand(i + 1).WriteArray(hr[i])

    def down(src, sx, sy, alg='average'):
        bnd = (bx0 - sx * LR, by0 + sy * LR, bx1 - sx * LR, by1 + sy * LR)
        return gdal.Warp('', src, format='MEM', dstSRS='EPSG:32611', outputBounds=bnd, xRes=LR, yRes=LR,
                         resampleAlg=alg).ReadAsArray().astype(np.float64).mean(0)

    ref = down(mem, 0, 0)
    rows = []
    for s in (0.0, 0.25, 0.5):
        for ax in ('x', 'y'):
            b = down(mem, s if ax == 'x' else 0, s if ax == 'y' else 0)
            p, e = pc(ref, b), ecc(ref, b)
            rows.append({'chip_id': cid, 'test': 'shift', 'axis': ax, 's': s,
                         'pc_dx': p[0], 'pc_dy': p[1], 'ecc_dx': e[0], 'ecc_dy': e[1]})
    # 축소 방식 비교(이동 0): 기준 = 0.3 m HR의 average(=4x4 평균)
    for name, src, alg in (('src4326_average', str(W / 'hr_src.vrt'), 'average'),
                           ('src4326_cubic', str(W / 'hr_src.vrt'), 'cubic')):
        b = down(src, 0, 0, alg)
        p, e = pc(ref, b), ecc(ref, b)
        rows.append({'chip_id': cid, 'test': name, 'axis': '-', 's': 0.0,
                     'pc_dx': p[0], 'pc_dy': p[1], 'ecc_dx': e[0], 'ecc_dy': e[1]})
    return rows


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    chips = gpd.read_file(W / 'chips.geojson')
    with rasterio.open(W / 'lr_utm.vrt') as ls:
        lrt = (ls.transform.c, ls.transform.f)
    jobs = [(c.chip_id, *c.geometry.buffer(-6).bounds, lrt) for c in chips.itertuples()]
    with ProcessPoolExecutor(12) as ex:
        df = pd.DataFrame(list(ex.map(chip_job, jobs, chunksize=4)))
        df.to_csv(OUT / 'chips.csv', index=False)
        syn = pd.DataFrame([r for rs in ex.map(synth_job, jobs[::29]) for r in rs])
        syn.to_csv(OUT / 'synth.csv', index=False)
    print('done', len(df), len(syn))
