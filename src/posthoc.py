"""사후 분석(주 결론 아님, docs/frozen_eval.md 재실행 기록). 탐지기·평가 설정은 frozen_eval.md 그대로 쓰고 결과는 results/posthoc/.
1. S0 대비 이득: 각 방법 AP50 − S0 AP50, test 블록 부트스트랩(geo_eval.py와 같은 seed·같은 표본) 95% CI.
2. 색 보정 대조군: train 블록 LR(MS)–HR(PS, 1.2 m 평균 축소) 타일 쌍의 평가 영역 마스크 안 유효 픽셀에서 밴드별 선형 변환
   (gain, offset)을 최소제곱으로 학습해 S0·S3 test 출력에 적용(S0_color, S3_color, 무효 영역은 HR 채움값) → D_HR 탐지 → 병합
   → 지리 평가. S6·LR·원래 방법과의 AP50 차이와 CI.
3. 면적 구간별 재현율(CI)과 운영 conf 기준 거짓 양성 수(모든 방법).
4. 화질 지표–AP50 Spearman: SR 8개 방법과 S6 제외 7개.
5. 정성 그림: 건물 밀도가 중간인 test 구역 2곳, HR·LR·S0·S3·S4·S7·S6 패널에 GT(노랑), 운영 conf 탐지(TP 초록, FP 빨강).
제외 폭은 주 값(eval_buffer_lr)만 쓴다."""
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor

import cv2
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import spearmanr
from ultralytics import YOLO

import gdal_ops as g
from det_infer import rows
from geo_eval import BINS, EDGES, area, block_of, gt_in, load_pred, match, stats
from merge import merge

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = g.ROOT / 'results/posthoc'
PW = g.W / 'posthoc'
COLOR = {'S0_color': 'S0_gdal_cubic', 'S3_color': 'S3_realesrnet_x4plus'}
BUF = g.CFG['eval_buffer_lr']


def frozen_sha(weights):
    """frozen_eval.md에 기록한 SHA256과 같은 가중치인지 확인한다."""
    name = 'D_HR' if 'D_HR' in str(weights) else 'D_LR'
    want = re.search(rf'\| {name} SHA256 \| ([0-9a-f]{{64}}) \|', (g.ROOT / 'docs/frozen_eval.md').read_text()).group(1)
    got = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert got == want, (weights, got, want)
    return got


def fit_tile(args):
    """타일 하나의 밴드별 최소제곱 누적합(n, Σx, Σy, Σxx, Σxy, Σyy). x = LR 8비트, y = HR 8비트 4×4 평균."""
    tid, region = args
    with rasterio.open(g.W / f'tiles/lr/images/train/{tid}.tif') as s:
        lr, tr = s.read().astype(np.float64), s.transform
    with rasterio.open(g.W / f'tiles/hr/images/train/{tid}.tif') as s:
        hr = s.read().astype(np.float64)
    hr = hr.reshape(3, lr.shape[1], 4, lr.shape[2], 4).mean(axis=(2, 4))
    m = g.tile_mask(tr, lr.shape[1:], region)
    x, y = lr[:, m], hr[:, m]
    return np.stack([np.full(3, m.sum()), x.sum(1), y.sum(1), (x * x).sum(1), (x * y).sum(1), (y * y).sum(1)])


def color_fit(tiles, fp):
    region = fp.buffer(-BUF * g.LR)
    ids = sorted(tiles[(tiles.split == 'train') & tiles.use].tile_id)
    with ProcessPoolExecutor(g.CFG['n_proc']) as ex:
        n, sx, sy, sxx, sxy, syy = sum(ex.map(fit_tile, [(i, region) for i in ids]))
    a = (n * sxy - sx * sy) / (n * sxx - sx**2)
    b = (sy - a * sx) / n
    sse = syy - 2 * a * sxy - 2 * b * sy + a * a * sxx + 2 * a * b * sx + n * b * b
    r2 = 1 - sse / (syy - sy**2 / n)
    fit = {'bands': ['R', 'G', 'B'], 'gain': a.tolist(), 'offset': b.tolist(), 'r2': r2.tolist(), 'n_px_per_band': int(n[0]),
           'n_train_tiles': len(ids), 'mask': f'footprint inset {BUF} LR px, train tiles (LR grid)',
           'x': 'LR 8-bit (MS)', 'y': 'HR 8-bit (PS-RGB) 4x4 mean'}
    (OUT / 'color_fit.json').write_text(json.dumps(fit, indent=1))
    print('color fit', fit)
    return a, b


def make_color(tiles, fp, a, b):
    hr_fill, _ = g.fills()
    ids = sorted(tiles[(tiles.split == 'test') & tiles.use].tile_id)
    for name, src in COLOR.items():
        for i in ids:
            with rasterio.open(g.W / 'sr' / src / f'images/test/{i}.tif') as s:
                img, tr = s.read().astype(np.float64), s.transform
            out = np.clip(np.rint(a[:, None, None] * img + b[:, None, None]), 0, 255).astype(np.uint8)
            g.write_tif(PW / 'sr' / name / f'images/test/{i}.tif', g.filled(out, g.tile_mask(tr, out.shape[1:], fp), hr_fill), tr)
        print(f'{name}: {len(ids)} tiles from {src}')
    return ids


def detect(ids, tiles, te):
    """frozen 설정(D_HR, conf det_conf_ap, max_det, imgsz_hr, 병합 NMS IoU)으로 색 보정 행만 새로 탐지·병합한다."""
    cal = json.loads((g.ROOT / 'results/det_calibration.json').read_text())
    max_det = json.loads((g.W / 'label_stats.json').read_text())['max_det']
    weights = g.ROOT / 'runs' / g.W.name / 'D_HR/weights/best.pt'
    sha = frozen_sha(weights)
    model = YOLO(weights)
    (PW / 'preds').mkdir(parents=True, exist_ok=True)
    for name in COLOR:
        img_dir = PW / 'sr' / name / 'images/test'
        recs = []
        for r in model.predict([str(img_dir / f'{i}.tif') for i in ids], imgsz=g.CFG['imgsz_hr'], conf=g.CFG['det_conf_ap'],
                               max_det=max_det, stream=True, verbose=False):
            tid = r.path.rsplit('/', 1)[1][:-4]
            for c, p in zip(r.obb.conf.cpu().numpy(), r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 8)):
                recs.append([tid, float(c), *map(float, p)])
        cols = ['tile_id', 'conf'] + [f'{a}{k}' for k in range(1, 5) for a in ('x', 'y')]
        p = pd.DataFrame(recs, columns=cols)
        p.to_csv(PW / 'preds' / f'{name}.csv', index=False)
        per_tile = p.groupby('tile_id').size()
        assert per_tile.max() < max_det, (name, per_tile.max(), max_det)
        merge(p, img_dir, tiles, te, cal['nms_iou']).to_file(PW / 'preds' / f'{name}.geojson', driver='GeoJSON')
        print(f'{name}: {len(p)} tile predictions, max per tile {per_tile.max()} (max_det {max_det})')
    return {'D_HR_sha256': sha, 'max_det': max_det, 'nms_iou': cal['nms_iou'], 'conf_op': cal['conf_op']}


def eval_row(args):
    """행 하나: 매칭, 전체 표본 통계, 부트스트랩 통계(블록 가중치 행렬 W_boot)."""
    n, path, conf_op, region, gt_obb, g_blk, g_bin, test, W_boot = args
    p = load_pred(path, region)
    tp, g_conf = match(p, gt_obb)
    d = {'conf': p.conf.to_numpy(), 'tp': tp, 'p_blk': block_of(p.geometry, test), 'g_blk': g_blk, 'g_conf': g_conf,
         'g_bin': g_bin, 'conf_op': conf_op}
    op = d['conf'] >= conf_op
    full = stats(d, np.ones(len(test))) | {'false_positives': int((op & ~tp).sum()), 'n_pred_op': int(op.sum())}
    boot = pd.DataFrame([stats(d, w) for w in W_boot])
    keep = p.assign(tp=tp)[op]
    return n, full, boot, keep


def ci(x):
    return np.nanpercentile(x, [2.5, 97.5])


def evaluate(names, paths, conf_op):
    test, union, fp, bld = area('test')
    region, gt, gt_obb = gt_in(bld, fp, union, BUF)
    rng = np.random.default_rng(g.CFG['seed'])  # geo_eval.py main과 같은 표본
    boot = rng.integers(0, len(test), size=(g.CFG['bootstrap'], len(test)))
    W_boot = np.stack([np.bincount(b, minlength=len(test)) for b in boot]).astype(float)
    g_blk, g_bin = block_of(gt.geometry, test), np.digitize(gt.area_m2, EDGES[1:-1])
    jobs = [(n, paths[n], conf_op[n], region, gt_obb, g_blk, g_bin, test, W_boot) for n in names]
    with ProcessPoolExecutor(min(len(jobs), g.CFG['n_proc'])) as ex:
        res = {n: (f, b, k) for n, f, b, k in ex.map(eval_row, jobs)}
    return res, len(gt), region, gt


def tables(names, res, n_gt, conf_op):
    full = {n: res[n][0] for n in names}
    bs = {n: res[n][1] for n in names}
    # frozen 평가(results/geo_eval.csv, 같은 제외 폭)와 같은 값이 나오는지 확인한다.
    ge = pd.read_csv(g.ROOT / 'results/geo_eval.csv')
    ge = ge[ge.buffer_lr == BUF].set_index('method')
    for n in ge.index:
        assert abs(ge.loc[n, 'AP50'] - full[n]['AP50']) < 1e-9, (n, ge.loc[n, 'AP50'], full[n]['AP50'])
        assert np.allclose(ci(bs[n].AP50), ge.loc[n, ['AP50_ci_low', 'AP50_ci_high']].to_numpy().astype(float)), n

    def diff(n, ref):
        lo, hi = ci(bs[n].AP50 - bs[ref].AP50)
        return {f'AP50_minus_{ref}': full[n]['AP50'] - full[ref]['AP50'], 'diff_ci_low': lo, 'diff_ci_high': hi,
                'verdict': 'better' if lo > 0 else 'worse' if hi < 0 else 'inconclusive'}

    def ap_row(n):
        lo, hi = ci(bs[n].AP50)
        return {'method': n, 'AP50_geo': full[n]['AP50'], 'AP50_ci_low': lo, 'AP50_ci_high': hi}

    t1 = pd.DataFrame([ap_row(n) | diff(n, 'S0_gdal_cubic') for n in names if n not in ('HR', 'LR', 'S0_gdal_cubic')])
    t1.to_csv(OUT / 'gain_vs_S0.csv', index=False)
    t2 = []
    for n, src in COLOR.items():
        for ref in ('S6_realesrgan_ft_spacenet', 'LR', src):
            t2.append(ap_row(n) | {'reference': ref, 'reference_AP50': full[ref]['AP50']}
                      | {k.replace(f'_minus_{ref}', '_minus_ref'): v for k, v in diff(n, ref).items()})
    t2 = pd.DataFrame(t2)
    t2.to_csv(OUT / 'color_control.csv', index=False)
    t3 = []
    for n in names:
        r = {'method': n, 'conf_op': conf_op[n], 'n_gt': n_gt}
        for b in BINS:
            r[f'recall_{b}'] = full[n][f'recall_{b}']
            r[f'recall_{b}_ci_low'], r[f'recall_{b}_ci_high'] = ci(bs[n][f'recall_{b}'])
        r['false_positives'], r['n_pred_op'] = full[n]['false_positives'], full[n]['n_pred_op']
        t3.append(r)
    t3 = pd.DataFrame(t3)
    t3.to_csv(OUT / 'area_recall_fp.csv', index=False)
    return t1, t2, t3


def spearman_table():
    s = pd.read_csv(g.ROOT / 'results/summary.csv')
    sr = s[s.path.str.startswith('path2')]
    out = []
    for k, sign in (('PSNR', 1), ('SSIM', 1), ('LPIPS', -1), ('edge_precision', 1)):
        for label, d in (('all SR (n=8)', sr), ('excl. S6 (n=7)', sr[sr.method != 'S6_realesrgan_ft_spacenet'])):
            rho, p = spearmanr(sign * d[k], d.AP50_geo)
            out.append({'metric': k, 'higher_is_better': sign == 1, 'set': label, 'n_methods': len(d),
                        'spearman_rho_vs_AP50': rho, 'p_value': p})
    t = pd.DataFrame(out)
    t.to_csv(OUT / 'spearman_excl_S6.csv', index=False)
    return t


def qualitative(res, region, gt):
    """건물 밀도가 중간인 구역 2곳: test 타일 중앙 153.6 m(HR 512 px) 창 중 평가 영역이 90% 이상인 창에서 GT 수가
    중앙값에 가장 가까운 창 2개(서로 다른 블록). TP/FP는 지리 평가 매칭(운영 conf, 제외 폭 주 값) 결과다."""
    tiles = gpd.read_file(g.W / 'tiles.geojson')
    test = tiles[(tiles.split == 'test') & tiles.use].reset_index(drop=True)
    half = 76.8
    cand = []
    for t in test.itertuples():
        x0, y0, x1, y1 = t.geometry.bounds
        cx, cy = g.LR * round((x0 + x1) / 2 / g.LR), g.LR * round((y0 + y1) / 2 / g.LR)
        w = gpd.GeoSeries.from_xy([cx], [cy]).buffer(half, cap_style='square').iloc[0]
        if w.intersection(region).area / w.area < 0.9:
            continue
        cand.append({'tile_id': t.tile_id, 'block_id': t.block_id, 'geom': w, 'n_gt': int(gt.centroid.within(w).sum())})
    cand = pd.DataFrame(cand)
    med = cand.n_gt.median()
    cand = cand.assign(dist=(cand.n_gt - med).abs()).sort_values(['dist', 'tile_id'])
    picks = cand.drop_duplicates('block_id').head(2)
    panels = ['HR', 'LR', 'S0_gdal_cubic', 'S3_realesrnet_x4plus', 'S4_realesrgan_x4plus', 'S7_osediff_x4',
              'S6_realesrgan_ft_spacenet']
    paths = []
    for c in picks.itertuples():
        x0, y0, x1, y1 = c.geom.bounds
        fig, axs = plt.subplots(2, 4, figsize=(16, 8.4), constrained_layout=True)
        for ax in axs.ravel():
            ax.axis('off')
        for ax, n in zip(axs.ravel(), panels):
            kind = {'HR': 'tiles/hr', 'LR': 'tiles/lr'}.get(n, f'sr/{n}')
            with rasterio.open(g.W / kind / f'images/test/{c.tile_id}.tif') as s:
                im = s.read(window=g.window(s, c.geom)).transpose(1, 2, 0)
            ax.imshow(cv2.resize(im, (512, 512), interpolation=cv2.INTER_NEAREST), extent=(x0, x1, y0, y1))
            for gm in gt.geometry[gt.intersects(c.geom)]:
                for part in getattr(gm, 'geoms', [gm]):
                    ax.plot(*part.exterior.xy, color='#ffd400', lw=1.2)
            k = res[n][2]
            k = k[k.intersects(c.geom)]
            for gm, ok in zip(k.geometry, k.tp):
                ax.plot(*gm.exterior.xy, color='#19c37d' if ok else '#ff3b30', lw=1.4)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_title(f'{n.split("_")[0]}  TP {int(k.tp.sum())}  FP {int((~k.tp).sum())}', fontsize=11)
        leg = axs.ravel()[-1]
        for col, lab in (('#ffd400', 'GT building'), ('#19c37d', 'TP (geo match, IoU >= 0.5)'), ('#ff3b30', 'FP')):
            leg.plot([], [], color=col, lw=2, label=lab)
        leg.legend(loc='center', frameon=False, fontsize=11,
                   title=f'tile {c.tile_id}, 153.6 m window\n{c.n_gt} GT (median window {med:.0f})\n'
                         'operating conf: D_HR / D_LR (frozen)\npost-hoc, not a main result')
        p = g.FIG / f'qualitative_{c.tile_id}.png'
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(str(p.relative_to(g.ROOT)))
        print('PNG', p)
    return paths


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = gpd.read_file(g.W / 'tiles.geojson')
    fp = gpd.read_file(g.W / 'footprint.geojson').geometry.iloc[0]
    te = json.loads((g.W / 'extent.json').read_text())['te']
    a, b = color_fit(tiles, fp)
    ids = make_color(tiles, fp, a, b)
    frozen = detect(ids, tiles.set_index('tile_id'), te)
    names = [r[0] for r in rows('test')] + list(COLOR)
    paths = {n: g.W / 'preds/test' / f'{n}.geojson' for n in names if n not in COLOR}
    paths |= {n: PW / 'preds' / f'{n}.geojson' for n in COLOR}
    conf_op = {n: frozen['conf_op']['D_LR' if n == 'LR' else 'D_HR'] for n in names}
    res, n_gt, region, gt = evaluate(names, paths, conf_op)
    t1, t2, t3 = tables(names, res, n_gt, conf_op)
    t4 = spearman_table()
    figs = qualitative(res, region, gt)
    info = {'note': 'post-hoc analysis, not a main result (docs/frozen_eval.md)', 'buffer_lr': BUF, 'n_gt': n_gt,
            'bootstrap': g.CFG['bootstrap'], 'seed': g.CFG['seed'], **frozen, 'figures': figs}
    (OUT / 'posthoc.json').write_text(json.dumps(info, indent=1))
    pd.set_option('display.width', 250)
    for t in (t1, t2, t3, t4):
        print(t.round(4).to_string())


if __name__ == '__main__':
    main()
