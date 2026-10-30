"""G1~G8: GDAL 점검, 모자이크 VRT, UTM 기준 격자, 공간 분할, 정합 검사, 8비트 변환, 보간 SR, SR GeoTIFF 저장."""
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from osgeo import gdal, osr
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'configs/default.yaml').read_text())
D = ROOT / 'data'
RAW = D / 'raw'
W = D / ('smoke' if CFG['smoke'] else 'full')
FIG = ROOT / 'results/figures'
BIN = Path(sys.executable).parent
UTM = 'EPSG:32611'
HR, LR, B = CFG['hr_res'], CFG['lr_res'], CFG['block_m']
SPLITS = ('train', 'val', 'test')
gdal.UseExceptions()


def chip_path(kind, cid):
    ext = 'geojson' if kind == 'geojson_buildings' else 'tif'
    return RAW / kind / f'SN2_buildings_train_AOI_2_Vegas_{kind}_img{cid}.{ext}'


def window_ids():
    return [int(x) for x in (D / 'window_chips.txt').read_text().split()]


def run(cmd):
    print('$', ' '.join(map(str, cmd)), flush=True)
    subprocess.run([str(BIN / cmd[0])] + [str(c) for c in cmd[1:]], check=True)


def window(src, geom):
    """격자에 정확히 맞는 블록의 정수 픽셀 창."""
    x0, y0, x1, y1 = geom.bounds
    t = src.transform
    v = np.array([(x0 - t.c) / t.a, (t.f - y1) / -t.e, (x1 - x0) / t.a, (y1 - y0) / -t.e])
    assert np.allclose(v, np.rint(v), atol=1e-6), ('window not on grid', v)
    return Window(*np.rint(v).astype(int))


def g1():
    """샘플 칩 5장씩 gdalinfo -stats, 축별 화소 크기 비율, MS 밴드와 PS-RGB 밴드 상관."""
    ids = window_ids()
    sample = ids[:: len(ids) // 5][:5]
    to_utm = Transformer.from_crs('EPSG:4326', UTM, always_xy=True)
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    out = {'sample_chips': sample, 'chips': []}
    for cid in sample:
        rec = {'chip_id': cid}
        for kind in ('PS-RGB', 'MS'):
            info = gdal.Info(str(chip_path(kind, cid)), format='json', stats=True)
            gt = info['geoTransform']
            srs = osr.SpatialReference(info['coordinateSystem']['wkt'])
            srs.AutoIdentifyEPSG()
            lon, lat = info['cornerCoordinates']['center']
            x0, y0 = to_utm.transform(lon, lat)
            x1, _ = to_utm.transform(lon + gt[1], lat)
            _, y1 = to_utm.transform(lon, lat + gt[5])
            md = gdal.Info(str(chip_path(kind, cid)), format='json', options=['-mdd', 'all'])
            rec[kind] = {
                'crs': f"EPSG:{srs.GetAuthorityCode(None)}", 'size': info['size'], 'pixel_deg': [gt[1], gt[5]],
                'pixel_m_center': [x1 - x0, y0 - y1], 'bands': len(info['bands']),
                'dtype': sorted({b['type'] for b in info['bands']}),
                'nodata': [b.get('noDataValue') for b in info['bands']],
                'description': [b.get('description', '') for b in info['bands']],
                'color_interp': [b.get('colorInterpretation') for b in info['bands']],
                'nbits': md.get('metadata', {}).get('IMAGE_STRUCTURE', {}).get('NBITS'),
                'min': [b['minimum'] for b in info['bands']], 'max': [b['maximum'] for b in info['bands']],
                'mean': [round(b['mean'], 1) for b in info['bands']],
            }
        ps, ms = rec['PS-RGB'], rec['MS']
        rec['ratio_deg'] = [ms['pixel_deg'][0] / ps['pixel_deg'][0], ms['pixel_deg'][1] / ps['pixel_deg'][1]]
        rec['ratio_m'] = [ms['pixel_m_center'][0] / ps['pixel_m_center'][0], ms['pixel_m_center'][1] / ps['pixel_m_center'][1]]
        # PS-RGB를 MS 격자로 평균 축소해 밴드 간 상관을 잰다(LR RGB 밴드 선택 근거).
        msd = gdal.Open(str(chip_path('MS', cid)))
        g = msd.GetGeoTransform()
        bounds = (g[0], g[3] + g[5] * msd.RasterYSize, g[0] + g[1] * msd.RasterXSize, g[3])
        psd = gdal.Warp('', str(chip_path('PS-RGB', cid)), format='MEM', outputBounds=bounds,
                        width=msd.RasterXSize, height=msd.RasterYSize, resampleAlg='average')
        a = psd.ReadAsArray().reshape(3, -1).astype(np.float64)
        m = msd.ReadAsArray().reshape(msd.RasterCount, -1).astype(np.float64)
        rec['corr_ps_ms'] = np.corrcoef(np.vstack([a, m]))[:3, 3:].round(4).tolist()
        out['chips'].append(rec)
    corr = np.mean([r['corr_ps_ms'] for r in out['chips']], axis=0)
    out['corr_mean'] = corr.round(4).tolist()
    out['best_ms_band_for_ps_band'] = (corr.argmax(axis=1) + 1).tolist()
    W.mkdir(parents=True, exist_ok=True)
    (W / 'g1.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('corr_mean', 'best_ms_band_for_ps_band')}))
    tol = CFG['g1_ratio_tol']
    for r in out['chips']:
        print(r['chip_id'], 'ratio_deg', np.round(r['ratio_deg'], 4), 'ratio_m', np.round(r['ratio_m'], 4),
              'PS m', np.round(r['PS-RGB']['pixel_m_center'], 4), 'MS m', np.round(r['MS']['pixel_m_center'], 4))
        for v in r['ratio_deg'] + r['ratio_m']:
            assert abs(v / 4 - 1) <= tol, f"G1: MS/PS-RGB pixel ratio {v:.4f} is not ~4 (chip {r['chip_id']})"
    assert out['best_ms_band_for_ps_band'] == CFG['ms_rgb_bands'], \
        f"G1: ms_rgb_bands {CFG['ms_rgb_bands']} disagree with correlation {out['best_ms_band_for_ps_band']}"


def g2():
    """칩 모자이크 VRT: hr_src.vrt(PS-RGB), ms_src.vrt(MS, 동서 화소 크기·원점 메타데이터 정정)."""
    W.mkdir(parents=True, exist_ok=True)
    # MS 칩 geotransform은 163 px를 칩 범위에 맞춘 동서 화소 크기를 적었지만 자료 간격은 4×PS다(ECC 아핀 추정
    # 1.080065e-5° vs 1.08e-5°, +0.006%). 원래 MS 격자 원점을 전역 칩 격자 서쪽 끝으로 보면 칩 열 k의 원점은
    # 650k PS px = 162.5k MS px라 홀수 열 칩의 MS 원점은 칩 원점보다 2 PS px 서쪽이다(docs/decisions.md).
    # 동서만 정정하고 남북은 그대로 둔다(-a_ullr VRT, 픽셀값 불변).
    idx = pd.read_csv(D / 'chip_index.csv')
    step = idx.maxx.iloc[0] - idx.minx.iloc[0]
    col = dict(zip(idx.chip_id, np.rint((idx.minx - idx.minx.min()) / step).astype(int)))
    fix = W / 'ms_fix'
    fix.mkdir(exist_ok=True)
    for c in window_ids():
        ps, ms = gdal.Open(str(chip_path('PS-RGB', c))), gdal.Open(str(chip_path('MS', c)))
        gp, gm = ps.GetGeoTransform(), ms.GetGeoTransform()
        assert (gp[0], gp[3]) == (gm[0], gm[3]) and gp[2] == gp[4] == gm[2] == gm[4] == 0, (c, gp, gm)
        x0 = gm[0] - 2 * gp[1] * (col[c] % 2)
        ullr = (x0, gm[3], x0 + 4 * gp[1] * ms.RasterXSize, gm[3] + gm[5] * ms.RasterYSize)
        gdal.Translate(str(fix / f'{c}.vrt'), ms, format='VRT', outputBounds=ullr)
    for kind, name, files in (('PS-RGB', 'hr_src.vrt', [chip_path('PS-RGB', c) for c in window_ids()]),
                              ('MS', 'ms_src.vrt', [fix / f'{c}.vrt' for c in window_ids()])):
        lst = W / f'{name}.txt'
        lst.write_text('\n'.join(map(str, files)) + '\n')
        run(['gdalbuildvrt', '-overwrite', '-input_file_list', lst, W / name])


def footprints():
    df = pd.read_csv(D / 'chip_index.csv').set_index('chip_id').loc[window_ids()]
    geoms = [box(r.minx, r.miny, r.maxx, r.maxy).segmentize(1e-5) for r in df.itertuples()]
    return gpd.GeoDataFrame({'chip_id': df.index}, geometry=geoms, crs='EPSG:4326').to_crs(UTM)


def g3():
    """train 칩 footprint 합집합(UTM)과 HR 0.3 m / LR 1.2 m 공통 격자 VRT."""
    chips = footprints()
    chips.to_file(W / 'chips.geojson', driver='GeoJSON')
    fp = unary_union(chips.geometry)
    gpd.GeoDataFrame(geometry=[fp], crs=UTM).to_file(W / 'footprint.geojson', driver='GeoJSON')
    x0, y0, x1, y1 = fp.bounds
    te = [math.ceil(x0 / LR) * LR, math.ceil(y0 / LR) * LR, math.floor(x1 / LR) * LR, math.floor(y1 / LR) * LR]
    (W / 'extent.json').write_text(json.dumps({'te': te}))
    common = ['gdalwarp', '-overwrite', '-of', 'VRT', '-t_srs', UTM, '-te', *te, '-r', 'cubic',
              '-multi', '-wo', f"NUM_THREADS={CFG['gdal_threads']}"]
    run(common + ['-tr', HR, HR, W / 'hr_src.vrt', W / 'hr_utm.vrt'])
    bands = [x for b in CFG['ms_rgb_bands'] for x in ('-b', b)]
    run(common + ['-tr', LR, LR, *bands, W / 'ms_src.vrt', W / 'lr_utm.vrt'])
    with rasterio.open(W / 'hr_utm.vrt') as h, rasterio.open(W / 'lr_utm.vrt') as lr:
        assert h.crs == lr.crs and h.count == lr.count == 3
        assert np.allclose(h.res, (HR, HR)) and np.allclose(lr.res, (LR, LR)), (h.res, lr.res)
        assert np.allclose((h.transform.c, h.transform.f), (lr.transform.c, lr.transform.f), atol=1e-6)
        assert (h.width, h.height) == (lr.width * 4, lr.height * 4), (h.shape, lr.shape)
        print('G3 grid OK: LR', lr.shape, 'x 4 = HR', h.shape, 'origin', (lr.transform.c, lr.transform.f))


def g4():
    """576 m 블록 격자, 건물 수 기준 70/15/15 분할(splits.geojson), 건물 UTM 변환."""
    te = json.loads((W / 'extent.json').read_text())['te']
    fp = gpd.read_file(W / 'footprint.geojson').geometry.iloc[0]
    chips = gpd.read_file(W / 'chips.geojson')
    frames = []
    for cid in window_ids():
        g = gpd.read_file(chip_path('geojson_buildings', cid))
        if len(g):
            assert g.crs.to_epsg() == 4326, (cid, g.crs)
            frames.append(g.assign(chip_id=cid))
    bld = pd.concat(frames, ignore_index=True)
    bld = bld[~(bld.geometry.isna() | bld.geometry.is_empty)].to_crs(UTM)
    bld['area_m2'] = bld.geometry.area
    bld.to_file(W / 'buildings_utm.geojson', driver='GeoJSON')
    nx, ny = int((te[2] - te[0] + 1e-6) // B), int((te[3] - te[1] + 1e-6) // B)
    rows = []
    for r in range(ny):
        for c in range(nx):
            bx = box(te[0] + c * B, te[3] - (r + 1) * B, te[0] + (c + 1) * B, te[3] - r * B)
            rows.append({'row': r, 'col': c, 'coverage': bx.intersection(fp).area / B**2, 'geometry': bx})
    blocks = gpd.GeoDataFrame(rows, crs=UTM)
    edge_km2 = ((te[2] - te[0]) * (te[3] - te[1]) - len(blocks) * B**2) / 1e6
    lost_fp_km2 = (fp.area - fp.intersection(unary_union(blocks.geometry)).area) / 1e6
    blocks = blocks[blocks.coverage > 0].reset_index(drop=True)
    blocks['block_id'] = np.arange(len(blocks))
    cent = gpd.GeoDataFrame(geometry=bld.geometry.centroid, crs=UTM)
    blocks['n_buildings'] = gpd.sjoin(cent, blocks, predicate='within').groupby('index_right').size() \
        .reindex(blocks.index, fill_value=0).values
    blocks['n_chips'] = gpd.sjoin(blocks, chips, predicate='intersects').groupby(level=0).size() \
        .reindex(blocks.index, fill_value=0).values
    # 시드 고정으로 블록을 섞고, 목표 대비 건물 수가 가장 부족한 분할에 차례로 배정한다.
    order = np.random.default_rng(CFG['seed']).permutation(len(blocks))
    target = np.array(CFG['split_ratio']) * blocks.n_buildings.sum()
    got = np.zeros(3)
    split = np.empty(len(blocks), dtype=object)
    for i in order:
        k = int(np.argmin(got / target))
        split[i] = SPLITS[k]
        got[k] += blocks.n_buildings[i]
    blocks['split'] = split
    assert all((blocks.split == s).any() for s in SPLITS)
    blocks['use'] = True
    if CFG['smoke']:
        # 분할마다 칩 수가 smoke_chips/3에 가장 가까운 블록 1개(동률이면 커버율 높은 쪽).
        blocks['use'] = False
        for s in SPLITS:
            cand = blocks[blocks.split == s].assign(d=lambda t: (t.n_chips - CFG['smoke_chips'] / 3).abs())
            blocks.loc[cand.sort_values(['d', 'coverage'], ascending=[True, False]).index[0], 'use'] = True
    blocks.to_file(W / 'splits.geojson', driver='GeoJSON')
    summ = blocks[blocks.use].assign(valid_km2=lambda t: t.coverage * B**2 / 1e6).groupby('split')[
        ['block_id', 'n_buildings', 'valid_km2', 'n_chips']].agg({'block_id': 'count', 'n_buildings': 'sum',
                                                                 'valid_km2': 'sum', 'n_chips': 'sum'})
    summ = summ.reindex(SPLITS).rename(columns={'block_id': 'blocks'})
    for col in ('blocks', 'n_buildings', 'valid_km2'):
        summ[col + '_ratio'] = (summ[col] / summ[col].sum()).round(3)
    summ.to_csv(W / 'split_summary.csv')
    print(summ.to_string())
    print(f'grid {ny}x{nx} blocks, kept {len(blocks)} (coverage>0); edge area dropped {edge_km2:.3f} km2, '
          f'footprint outside kept blocks {lost_fp_km2:.3f} km2; buildings {len(bld)}')
    (W / 'g4.json').write_text(json.dumps({'grid': [ny, nx], 'kept_blocks': len(blocks), 'edge_dropped_km2': edge_km2,
                                           'footprint_lost_km2': lost_fp_km2, 'buildings_total': len(bld)}))


def used_blocks():
    b = gpd.read_file(W / 'splits.geojson')
    return b[b.use].reset_index(drop=True)


def even_dft(n):
    """phaseCorrelate는 getOptimalDFTSize로 늘린 크기가 홀수인 축에서 +0.5 px 편향이 생긴다. n 이하 짝수 최적 크기."""
    while n % 2 or cv2.getOptimalDFTSize(n) != n:
        n -= 1
    return n


def norm(x):
    return ((x - x.mean()) / x.std()).astype(np.float32)


def g5_block(args):
    geom, wins = args
    x0, _, _, y1 = geom.bounds
    with rasterio.open(W / 'hr_utm.vrt') as h, rasterio.open(W / 'lr_utm.vrt') as lr:
        hr = h.read(window=window(h, geom)).astype(np.float64)
        lo = lr.read(window=window(lr, geom)).astype(np.float64)
    n = lo.shape[1]
    a = hr.reshape(3, n, 4, n, 4).mean(axis=(2, 4)).mean(axis=0)
    b = lo.mean(axis=0)
    res = []
    for wx0, wy0, wx1, wy1 in wins:
        c0, c1 = math.ceil((wx0 - x0) / LR), math.floor((wx1 - x0) / LR)
        r0, r1 = math.ceil((y1 - wy1) / LR), math.floor((y1 - wy0) / LR)
        if min(c1 - c0, r1 - r0) < CFG['g5_min_window']:
            continue
        w, h = even_dft(c1 - c0), even_dft(r1 - r0)
        wa, wb = np.ascontiguousarray(a[r0:r0 + h, c0:c0 + w]), np.ascontiguousarray(b[r0:r0 + h, c0:c0 + w])
        (dx, dy), resp = cv2.phaseCorrelate(wa, wb, cv2.createHanningWindow((w, h), cv2.CV_64F))
        # 교차 확인: ECC 평행이동(template=HR 축소, input=LR). 부호는 phaseCorrelate와 같다(LR 내용의 +열·+행 이동).
        _, m = cv2.findTransformECC(norm(wa), norm(wb), np.eye(2, 3, dtype=np.float32), cv2.MOTION_TRANSLATION,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
        res.append((dx, dy, resp, float(m[0, 2]), float(m[1, 2])))
    return res


def g5():
    """블록별 HR(1.2 m 평균 축소)↔LR 밝기 phaseCorrelate(짝수 최적 DFT 크기 창) + ECC 교차 확인. 칩 내부 창만 사용."""
    blocks = used_blocks()
    chips = gpd.read_file(W / 'chips.geojson')
    jobs = []
    for bg in blocks.geometry:
        inner = [c.buffer(-CFG['g5_inset_m']).intersection(bg) for c in chips.geometry[chips.intersects(bg)]]
        jobs.append((bg, [g.bounds for g in inner if not g.is_empty]))
    with ProcessPoolExecutor(CFG['n_proc']) as ex:
        out = list(ex.map(g5_block, jobs))
    rows = []
    for (_, blk), res in zip(blocks.iterrows(), out):
        assert res, f"G5: block {blk.block_id} has no valid window"
        r = np.array(res)
        dx, dy = np.median(r[:, 0]), np.median(r[:, 1])
        ex, ey = np.median(r[:, 3]), np.median(r[:, 4])
        rows.append({'block_id': blk.block_id, 'split': blk.split, 'coverage': round(blk.coverage, 3), 'n_windows': len(r),
                     'dx': dx, 'dy': dy, 'shift': math.hypot(dx, dy),
                     'window_shift_max': float(np.hypot(r[:, 0], r[:, 1]).max()), 'response_median': float(np.median(r[:, 2])),
                     'ecc_dx': ex, 'ecc_dy': ey, 'ecc_shift': math.hypot(ex, ey)})
    df = pd.DataFrame(rows)
    df.to_csv(W / 'g5_registration.csv', index=False)
    print(df[['n_windows', 'dx', 'dy', 'shift', 'window_shift_max', 'response_median', 'ecc_dx', 'ecc_dy', 'ecc_shift']]
          .describe().round(4).to_string())
    bad = df[df['shift'] > CFG['g5_max_shift']]
    # train 블록 초과는 경고로 남기고(제외하지 않음), val·test 블록 초과는 멈춘다(사용자 결정, docs/decisions.md).
    print(f"G5 WARNING: {(bad.split == 'train').sum()} train blocks exceed {CFG['g5_max_shift']} LR px (kept):\n"
          f"{bad[bad.split == 'train'].to_string()}")
    stop = bad[bad.split != 'train']
    assert stop.empty, f"G5: {len(stop)} val/test blocks exceed {CFG['g5_max_shift']} LR px:\n{stop.to_string()}"


def block_read(path, geom, fp):
    with rasterio.open(path) as src:
        win = window(src, geom)
        img = src.read(window=win)
        mask = rasterize([fp.intersection(geom)], out_shape=img.shape[1:], transform=src.window_transform(win),
                         dtype=np.uint8).astype(bool)
    return img, mask


def g6_hist(args):
    path, geom, fp = args
    img, mask = block_read(path, geom, fp)
    return np.stack([np.bincount(b[mask], minlength=65536) for b in img])


def g6_sum(args):
    path, geom, fp = args
    img, mask = block_read(path, geom, fp)
    return np.array([b[mask].sum(dtype=np.int64) for b in img]), int(mask.sum())


def filled(img, mask, fill):
    return np.where(mask[None], img, np.asarray(fill, np.uint8)[:, None, None])


def g6():
    """train 블록 유효 픽셀 밴드별 2~98 백분위 → gdal_translate -ot Byte -scale_N (VRT), 채움값 = 8비트 유효 평균."""
    blocks = used_blocks()
    fp = gpd.read_file(W / 'footprint.geojson').geometry.iloc[0]
    train = blocks[blocks.split == 'train'].geometry
    rad = {}
    with ProcessPoolExecutor(CFG['n_proc']) as ex:
        for key, src, dst in (('hr', 'hr_utm.vrt', 'hr_8bit.vrt'), ('lr', 'lr_utm.vrt', 'lr_8bit.vrt')):
            hist = sum(ex.map(g6_hist, [(W / src, g, fp) for g in train]))
            cdf = hist.cumsum(axis=1) / hist.sum(axis=1, keepdims=True)
            lo, hi = [[int(np.searchsorted(c, p / 100)) for c in cdf] for p in CFG['pct']]
            scale = [x for i in range(3) for x in (f'-scale_{i + 1}', lo[i], hi[i], 0, 255)]
            run(['gdal_translate', '-of', 'VRT', '-ot', 'Byte', *scale, W / src, W / dst])
            parts = list(ex.map(g6_sum, [(W / dst, g, fp) for g in train]))
            mean = sum(p[0] for p in parts) / sum(p[1] for p in parts)
            rad[key] = {'p_lo': lo, 'p_hi': hi, 'mean8': mean.round(3).tolist(), 'fill': np.rint(mean).astype(int).tolist(),
                        'n_valid_px': int(sum(p[1] for p in parts))}
    (W / 'radiometry.json').write_text(json.dumps(rad, indent=1))
    print(json.dumps(rad))
    # 색 비교 PNG: 커버율 중앙값 블록과 최소 블록. 왼쪽 HR(0.6 m로 축소), 오른쪽 LR(최근접 2배). 무효 영역은 채움값.
    FIG.mkdir(parents=True, exist_ok=True)
    srt = blocks.sort_values('coverage')
    for blk in (srt.iloc[len(srt) // 2], srt.iloc[0]):
        hr, hm = block_read(W / 'hr_8bit.vrt', blk.geometry, fp)
        lo8, lm = block_read(W / 'lr_8bit.vrt', blk.geometry, fp)
        h = cv2.resize(filled(hr, hm, rad['hr']['fill']).transpose(1, 2, 0), (960, 960), interpolation=cv2.INTER_AREA)
        lr = cv2.resize(filled(lo8, lm, rad['lr']['fill']).transpose(1, 2, 0), (960, 960), interpolation=cv2.INTER_NEAREST)
        png = FIG / f'g6_hr_lr_filled_block{blk.block_id}_{blk.split}_cov{blk.coverage:.2f}.png'
        cv2.imwrite(str(png), cv2.cvtColor(np.hstack([h, lr]), cv2.COLOR_RGB2BGR))
        print('PNG', png)


def fills():
    rad = json.loads((W / 'radiometry.json').read_text())
    return rad['hr']['fill'], rad['lr']['fill']


def write_tif(path, img, transform):
    """8비트 3밴드 GeoTIFF(DEFLATE, TILED)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', driver='GTiff', width=img.shape[2], height=img.shape[1], count=3, dtype='uint8',
                       crs=UTM, transform=transform, compress='deflate', tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(img)


def tile_mask(transform, shape, fp):
    return rasterize([fp], out_shape=shape, transform=transform, dtype=np.uint8).astype(bool)


def log_sr_time(method, seconds, km2, load_sec, seconds_total):
    """SR 처리 시간(모델 로딩 제외, 처리한 면적 기준), 로딩 시간, 전체 실행 시간을 W/sr_timing.csv에 방법별 한 행으로 남긴다."""
    f = W / 'sr_timing.csv'
    df = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=['method', 'seconds', 'km2', 'load_sec', 'seconds_total'])
    row = pd.DataFrame([{'method': method, 'seconds': round(seconds, 2), 'km2': km2,
                         'load_sec': np.round(load_sec, 2),
                         'seconds_total': round(seconds_total, 2)}])
    pd.concat([df[df.method != method], row]).to_csv(f, index=False)


def g7():
    """보간 SR(test 블록만): lr_8bit.vrt → 0.3 m cubic(S0)·lanczos(S1) 블록 GeoTIFF. 무효 영역은 HR 채움값(M3)."""
    blocks = used_blocks()
    test = blocks[blocks.split == 'test']
    fp = gpd.read_file(W / 'footprint.geojson').geometry.iloc[0]
    hr_fill, _ = fills()
    for method, alg in (('S0_gdal_cubic', 'cubic'), ('S1_gdal_lanczos', 'lanczos')):
        t0 = time.time()
        for blk in test.itertuples():
            out = W / 'sr_blocks' / method / f'block_{blk.block_id:03d}.tif'
            out.parent.mkdir(parents=True, exist_ok=True)
            run(['gdalwarp', '-overwrite', '-te', *blk.geometry.bounds, '-tr', HR, HR, '-r', alg, '-multi',
                 '-wo', f"NUM_THREADS={CFG['gdal_threads']}", '-co', 'COMPRESS=DEFLATE', '-co', 'TILED=YES',
                 W / 'lr_8bit.vrt', out])
            with rasterio.open(out, 'r+') as dst:
                assert np.allclose(dst.res, (HR, HR)) and dst.shape == (round(B / HR),) * 2, (dst.res, dst.shape)
                dst.write(filled(dst.read(), tile_mask(dst.transform, dst.shape, fp), hr_fill))
        # 보간은 불러올 모델이 없어 로딩 시간이 없다(NaN).
        sec = time.time() - t0
        log_sr_time(method, sec, len(test) * B**2 / 1e6, np.nan, sec)
    # HR / LR(최근접 4배) / S0 비교 PNG 3장: test 블록 안쪽·유효 영역 안쪽 60 m의 건물 중심 3곳(시드 고정), 120 m(400 HR px).
    bld = gpd.read_file(W / 'buildings_utm.geojson')
    cen = bld.geometry.centroid
    inner = unary_union([g.buffer(-60) for g in test.geometry]).intersection(fp.buffer(-60))
    pick = cen[cen.within(inner)].sample(3, random_state=CFG['seed'])
    for k, c in enumerate(pick):
        x0, y1 = LR * math.floor(c.x / LR) - 60, LR * math.floor(c.y / LR) + 60
        geom = box(x0, y1 - 120, x0 + 120, y1)
        hr, _ = block_read(W / 'hr_8bit.vrt', geom, fp)
        lo, _ = block_read(W / 'lr_8bit.vrt', geom, fp)
        blk = test[test.contains(c)].iloc[0]
        with rasterio.open(W / 'sr_blocks/S0_gdal_cubic' / f'block_{blk.block_id:03d}.tif') as src:
            s0 = src.read(window=window(src, geom))
        lr4 = np.repeat(np.repeat(lo, 4, axis=1), 4, axis=2)
        png = FIG / f'g7_hr_lr_s0_{k}_block{blk.block_id}.png'
        cv2.imwrite(str(png), cv2.cvtColor(np.hstack([x.transpose(1, 2, 0) for x in (hr, lr4, s0)]), cv2.COLOR_RGB2BGR))
        print('PNG', png)


def save_sr(img, lr_tile, hr_tile, out, fill, fp):
    """G8: 딥러닝 SR 결과(RGB, HxWx3) 저장. transform = LR 타일 transform × Affine.scale(1/4), 무효 영역은 채움값(M3).
    저장 후 같은 지리 창의 HR 타일과 transform·shape가 같은지 assert."""
    with rasterio.open(lr_tile) as lr:
        t = lr.transform * Affine.scale(1 / 4)
        assert img.shape == (lr.height * 4, lr.width * 4, 3), (img.shape, lr.shape)
    arr = img.transpose(2, 0, 1)
    write_tif(out, filled(arr, tile_mask(t, arr.shape[1:], fp), fill), t)
    with rasterio.open(out) as a, rasterio.open(hr_tile) as h:
        assert a.transform.almost_equals(h.transform, 1e-9) and a.shape == h.shape and a.crs == h.crs, \
            ('G8 grid mismatch', out, a.transform, h.transform, a.shape, h.shape)


if __name__ == '__main__':
    steps = {'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4, 'g5': g5, 'g6': g6, 'g7': g7}
    for s in sys.argv[1:]:
        steps[s]()
