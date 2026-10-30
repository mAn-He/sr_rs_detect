"""결과 리포트 뷰어(기존 결과물만 읽는다: summary.csv, geo_eval.csv, results/posthoc, 예측 CSV, SR·HR·LR GeoTIFF, GT 라벨).
새 탐지·SR 추론은 하지 않고 frozen_eval.md 설정과 summary.csv를 바꾸지 않는다.
- A. results/report/full/index.html + images/: test 타일 전체(이미지는 상대 경로)
- B. results/report/highlights.html: 대표 타일 12개, 이미지 base64 단일 파일(15 MB 이하가 되도록 화질을 낮춰 가며 만든다)
- results/report/highlights_tiles.csv: 대표 타일 선정 목록과 기준
타일 단위 P/R/F1·TP/FP/FN은 병합 전 타일 예측(운영 conf 이상)과 타일 라벨의 회전 IoU ≥ match_iou 탐욕 매칭으로 낸 참고값이다.
PSNR·LPIPS는 타일 안 평가 영역(제외 폭 eval_buffer_lr) 마스크 평균(M6과 같은 방식). 차이 지도 vmax는 S0~S7 전체 타일의
평가 영역 |SR−HR|(RGB 평균) 99번째 백분위 하나로 고정한다."""
import base64
import io
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from html import escape

import cv2
import geopandas as gpd
import lpips
import matplotlib
import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image
from scipy.stats import spearmanr
from shapely import STRtree
from shapely.geometry import Polygon

import gdal_ops as g
from geo_eval import BINS, area, gt_in

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = g.ROOT / 'results/report'
FULL = OUT / 'full'
POST = g.ROOT / 'results/posthoc'
PW = g.W / 'posthoc'
SR = list(g.CFG['sr_methods'])
POSTHOC = ['S0_color', 'S3_color'] if (POST / 'color_control.csv').exists() else []
METHODS = ['HR', 'LR', *SR, *POSTHOC]
N = 1024  # HR 타일 픽셀
P = 512   # 패널 픽셀
C = 256   # 확대 크롭 픽셀
CANNY = json.loads((g.ROOT / 'results/canny_thresholds.json').read_text())
CAL = json.loads((g.ROOT / 'results/det_calibration.json').read_text())
HL_LIMIT = 15e6
OFFICIAL = '공식 지표는 summary.csv(지리 병합·블록 부트스트랩)입니다. 타일 단위 P/R/F1·TP/FP/FN은 병합 전 타일 기준 참고값입니다.'


def short(m):
    return m if m in ('HR', 'LR') or m in POSTHOC else m.split('_')[0]


def image(m, tid):
    """HR 격자(1024) RGB. LR은 최근접 4배."""
    if m == 'LR':
        path = g.W / 'tiles/lr/images/test' / f'{tid}.tif'
    elif m == 'HR':
        path = g.W / 'tiles/hr/images/test' / f'{tid}.tif'
    elif m in POSTHOC:
        path = PW / 'sr' / m / 'images/test' / f'{tid}.tif'
    else:
        path = g.W / 'sr' / m / 'images/test' / f'{tid}.tif'
    with rasterio.open(path) as s:
        img = s.read().transpose(1, 2, 0)
    if m == 'LR':
        img = np.repeat(np.repeat(img, 4, axis=0), 4, axis=1)
    assert img.shape == (N, N, 3), (m, tid, img.shape)
    return np.ascontiguousarray(img)


def preds(m):
    path = PW / 'preds' / f'{m}.csv' if m in POSTHOC else g.W / 'preds/test' / f'{m}.csv'
    p = pd.read_csv(path)
    p = p[p.conf >= CAL['conf_op']['D_LR' if m == 'LR' else 'D_HR']]
    xy = p[[f'{a}{k}' for k in range(1, 5) for a in ('x', 'y')]].to_numpy() * (4 if m == 'LR' else 1)
    return {tid: (xy[idx], p.conf.to_numpy()[idx]) for tid, idx in p.groupby('tile_id').indices.items()}


def match(pxy, conf, gpoly):
    """conf 내림차순 탐욕 매칭(타일 픽셀 좌표). 반환: 예측별 TP 여부, GT별 매칭 여부."""
    gp = [Polygon(x.reshape(4, 2)) for x in gpoly]
    tree = STRtree(gp) if gp else None
    used = np.zeros(len(gp), bool)
    tp = np.zeros(len(pxy), bool)
    for i in np.argsort(-conf, kind='stable'):
        a = Polygon(pxy[i].reshape(4, 2))
        best, bj = g.CFG['match_iou'], -1
        for j in (tree.query(a, predicate='intersects') if tree else []):
            if used[j]:
                continue
            inter = a.intersection(gp[j]).area
            iou = inter / (a.area + gp[j].area - inter)
            if iou >= best:
                best, bj = iou, j
        if bj >= 0:
            used[bj], tp[i] = True, True
    return tp, used


def nan2none(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def tile_masks(tid, tiles, fp, region):
    t = tiles.loc[tid]
    with rasterio.open(g.W / 'tiles/hr/images/test' / f'{tid}.tif') as s:
        tr = s.transform
    return g.tile_mask(tr, (N, N), fp), g.tile_mask(tr, (N, N), region), t


def measure(ids, tiles, fp, region):
    """1차: 타일×방법 지표(PSNR, LPIPS, 타일 단위 매칭)와 |SR−HR| 히스토그램(S0~S7, 평가 영역)."""
    net = lpips.LPIPS(net='alex', spatial=True).cuda().eval()
    pr = {m: preds(m) for m in METHODS}
    hist = np.zeros(766, np.int64)
    out = []
    for tid in ids:
        valid, ev, _ = tile_masks(tid, tiles, fp, region)
        lab = np.loadtxt(g.W / 'tiles/hr/labels/test' / f'{tid}.txt', ndmin=2)
        gpoly = (lab[:, 1:] * N) if len(lab) else np.zeros((0, 8))
        hr = image('HR', tid)
        th = torch.from_numpy(hr.transpose(2, 0, 1).copy()).float().div(127.5).sub(1).unsqueeze(0).cuda()
        rec = {'id': tid, 'n_gt': len(gpoly), 'valid_frac': round(float(valid.mean()), 3),
               'gt': np.rint(gpoly).astype(int).tolist(), 'm': {}}
        for m in METHODS:
            r = {}
            if m not in ('HR', 'LR') and ev.any():
                x = image(m, tid)
                d = np.abs(hr.astype(np.int16) - x)
                r['psnr'] = 10 * np.log10(255**2 / (d[ev].astype(np.float64) ** 2).mean())
                with torch.no_grad():
                    tx = torch.from_numpy(x.transpose(2, 0, 1).copy()).float().div(127.5).sub(1).unsqueeze(0).cuda()
                    r['lpips'] = float(net(th, tx)[0, 0].cpu().numpy()[ev].mean())
                if m in SR:
                    hist += np.bincount(d.sum(axis=2)[ev].ravel(), minlength=766)
            pxy, conf = pr[m].get(tid, (np.zeros((0, 8)), np.zeros(0)))
            tp, used = match(pxy, conf, gpoly)
            ntp, nfp, nfn = int(tp.sum()), int((~tp).sum()), int((~used).sum())
            r |= {'TP': ntp, 'FP': nfp, 'FN': nfn,
                  'P': ntp / (ntp + nfp) if ntp + nfp else np.nan, 'R': ntp / (ntp + nfn) if ntp + nfn else np.nan,
                  'F1': 2 * ntp / (2 * ntp + nfp + nfn) if ntp + nfp + nfn else np.nan,
                  'det': [[*map(int, np.rint(q)), int(t)] for q, t in zip(pxy, tp)],
                  'fn': np.flatnonzero(~used).tolist()}
            rec['m'][m] = {k: (nan2none(v) if isinstance(v, float) else v) for k, v in r.items()}
        out.append(rec)
        print('measured', tid, flush=True)
    cdf = np.cumsum(hist) / hist.sum()
    vmax = float(np.searchsorted(cdf, 0.99)) / 3  # RGB 합 → 평균
    return out, vmax


def enc(arr, q, lossless=False):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, 'WEBP', quality=q, lossless=lossless, method=4)
    return b.getvalue()


def magma(x, vmax):
    lut = (plt.get_cmap('magma')(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    return lut[np.clip(x / vmax * 255, 0, 255).astype(np.uint8)]


def edges(img):
    return cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), CANNY['low'], CANNY['high']) > 0


def pool_max(e):
    return e.reshape(P, 2, P, 2).max(axis=(1, 3))


def edge_png(e, rgb):
    """엣지 1비트 투명 PNG(뷰어에서 HR 층 위에 SR 층을 screen 혼합으로 겹친다: 초록 + 빨강 = 노랑)."""
    p = Image.fromarray(e.astype(np.uint8), 'P')
    p.putpalette([0, 0, 0, *rgb])
    b = io.BytesIO()
    p.save(b, 'PNG', optimize=True, transparency=0, bits=1)
    return b.getvalue()


def render(args):
    """2차: 타일 하나의 모든 방법 이미지(키 → 바이트). crops=True면 원본 1024의 4×4 격자 256 크롭도 만든다.
    차이 지도는 diff_px 해상도(512 또는 LR 화소 평균인 256)로 저장한다."""
    tid, vmax, q_img, q_map, crops, diff_px, region, fp = args
    tiles = gpd.read_file(g.W / 'tiles.geojson').set_index('tile_id')
    valid, ev, _ = tile_masks(tid, tiles, fp, region)
    hr = image('HR', tid)
    s0 = image('S0_gdal_cubic', tid)
    out = {f'{tid}/edge_HR.png': edge_png(pool_max(edges(hr)), (40, 200, 70))}
    # 무효 영역 회색 빗금, 평가 제외(유효하지만 평가 영역 밖) 반투명
    ov = np.zeros((N, N, 4), np.uint8)
    yy, xx = np.mgrid[:N, :N]
    hatch = ((xx + yy) // 8) % 2 == 0
    ov[~valid] = (110, 110, 110, 120)
    ov[~valid & hatch] = (170, 170, 170, 220)
    ov[valid & ~ev] = (0, 0, 0, 110)
    b = io.BytesIO()
    Image.fromarray(ov).quantize(colors=4, method=Image.Quantize.FASTOCTREE).save(b, 'PNG', optimize=True)
    out[f'{tid}/mask.png'] = b.getvalue()
    for m in METHODS:
        x = hr if m == 'HR' else s0 if m == 'S0_gdal_cubic' else image(m, tid)
        out[f'{tid}/{m}_img.webp'] = enc(cv2.resize(x, (P, P), interpolation=cv2.INTER_AREA), q_img)
        if crops:
            for i in range(N // C):
                for j in range(N // C):
                    out[f'{tid}/{m}_c{i}{j}.webp'] = enc(np.ascontiguousarray(x[i * C:(i + 1) * C, j * C:(j + 1) * C]), q_img)
        if m != 'HR':
            d = np.abs(hr.astype(np.int16) - x).mean(axis=2).astype(np.float32)
            out[f'{tid}/{m}_hr.webp'] = enc(magma(cv2.resize(d, (diff_px,) * 2, interpolation=cv2.INTER_AREA), vmax), q_map)
        if m != 'S0_gdal_cubic':
            d = np.abs(s0.astype(np.int16) - x).mean(axis=2).astype(np.float32)
            out[f'{tid}/{m}_s0.webp'] = enc(magma(cv2.resize(d, (diff_px,) * 2, interpolation=cv2.INTER_AREA), vmax), q_map)
        if m != 'HR':
            out[f'{tid}/{m}_edge.png'] = edge_png(pool_max(edges(x)), (235, 60, 50))
    return tid, out


def charts():
    """개요 탭 matplotlib PNG(이름 → 바이트)와 S6 제외 Spearman."""
    s = pd.read_csv(g.ROOT / 'results/summary.csv')
    sr = s[s.path.str.startswith('path2')].reset_index(drop=True)
    col = {'useful': '#1f9d55', 'harmful': '#d64545', 'inconclusive': '#8a8a8a'}
    out, rho = {}, {}

    def png(fig, name):
        b = io.BytesIO()
        fig.savefig(b, format='png', dpi=110)
        plt.close(fig)
        out[name] = b.getvalue()

    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    y = np.arange(len(sr))[::-1]
    for yi, r in zip(y, sr.itertuples()):
        ax.errorbar(r.gain, yi, xerr=[[r.gain - r.gain_ci_low], [r.gain_ci_high - r.gain]], fmt='o', color=col[r.verdict],
                    capsize=3, ms=6)
    ax.axvline(0, color='#333', lw=1)
    ax.set_yticks(y, [short(m) for m in sr.method])
    ax.set_xlabel('gain = AP50(SR) − AP50(LR), 95% block-bootstrap CI')
    ax.set_title('(a) gain vs LR (path 1)')
    ax.grid(axis='x', color='#e5e5e5')
    png(fig, 'a_forest.png')
    for k, sign, name in (('PSNR', 1, 'b_psnr.png'), ('LPIPS', -1, 'b_lpips.png')):
        r_all = spearmanr(sign * sr[k], sr.AP50_geo)
        ex = sr[sr.method != 'S6_realesrgan_ft_spacenet']
        r_ex = spearmanr(sign * ex[k], ex.AP50_geo)
        rho[k] = (r_all, r_ex)
        fig, ax = plt.subplots(figsize=(6, 4.4), constrained_layout=True)
        ax.errorbar(sr[k], sr.AP50_geo, yerr=[sr.AP50_geo - sr.AP50_ci_low, sr.AP50_ci_high - sr.AP50_geo], fmt='o',
                    color='#2b6cb0', capsize=3)
        for r in sr.itertuples():
            ax.annotate(short(r.method), (getattr(r, k), r.AP50_geo), textcoords='offset points', xytext=(5, 4), fontsize=9)
        for m, ls in (('HR', '--'), ('LR', ':')):
            ax.axhline(s.set_index('method').loc[m, 'AP50_geo'], color='#555', ls=ls, lw=1, label=f'{m} AP50')
        ax.set_xlabel(f'{k} (evaluation mask){" — lower is better" if sign < 0 else ""}')
        ax.set_ylabel('AP50_geo (95% CI)')
        ax.set_title(f'(b) {k} vs AP50  ρ={r_all.statistic:.2f} (p={r_all.pvalue:.3f}, n=8); '
                     f'excl. S6 ρ={r_ex.statistic:.2f} (p={r_ex.pvalue:.3f}, n=7)', fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        png(fig, name)
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    w = 0.8 / len(s)
    cm = plt.get_cmap('tab10')
    for i, r in enumerate(s.itertuples()):
        v = np.array([getattr(r, f'recall_{b}') for b in BINS])
        lo = np.array([getattr(r, f'recall_{b}_ci_low') for b in BINS])
        hi = np.array([getattr(r, f'recall_{b}_ci_high') for b in BINS])
        ax.bar(np.arange(len(BINS)) + (i - (len(s) - 1) / 2) * w, v, w * 0.9, yerr=[v - lo, hi - v], color=cm(i % 10),
               label=short(r.method), error_kw={'lw': 0.7, 'ecolor': '#444'})
    ax.set_xticks(np.arange(len(BINS)), [f'{b} m²' for b in BINS])
    ax.set_ylabel('recall at operating conf (95% CI)')
    ax.set_title(f"(c) recall by building area  (operating conf D_HR {CAL['conf_op']['D_HR']:.3f}, "
                 f"D_LR {CAL['conf_op']['D_LR']:.3f})")
    ax.legend(ncol=5, fontsize=8, frameon=False)
    ax.grid(axis='y', color='#e5e5e5')
    png(fig, 'c_recall.png')
    return out, rho


def colorbar(vmax):
    fig, ax = plt.subplots(figsize=(4.2, 0.55), constrained_layout=True)
    fig.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, vmax), cmap='magma'), cax=ax,
                 orientation='horizontal', label='|difference| (8-bit, RGB mean)')
    b = io.BytesIO()
    fig.savefig(b, format='png', dpi=110)
    plt.close(fig)
    return b.getvalue()


def fmt_ci(v, lo, hi, d=3):
    return f'{v:.{d}f} <span class="ci">[{lo:.{d}f}, {hi:.{d}f}]</span>'


def overview_html(chart_src, rho):
    s = pd.read_csv(g.ROOT / 'results/summary.csv')
    s0 = pd.read_csv(POST / 'gain_vs_S0.csv').set_index('method') if (POST / 'gain_vs_S0.csv').exists() else None
    head = ['method', 'category', 'AP50_geo [CI]', 'gain vs LR [CI]'] + (['gain vs S0 [CI] (사후)'] if s0 is not None else []) \
        + ['recovery', 'verdict', 'PSNR', 'SSIM', 'LPIPS', 'FP']
    body = []
    for r in s.itertuples():
        sr = r.path.startswith('path2')
        cells = [escape(r.method), escape(r.category), fmt_ci(r.AP50_geo, r.AP50_ci_low, r.AP50_ci_high),
                 fmt_ci(r.gain, r.gain_ci_low, r.gain_ci_high) if sr else '–']
        if s0 is not None:
            cells.append(fmt_ci(s0.loc[r.method, 'AP50_minus_S0_gdal_cubic'], s0.loc[r.method, 'diff_ci_low'],
                                s0.loc[r.method, 'diff_ci_high']) if r.method in s0.index else '–')
        cells += [f'{r.recovery:.2f}' if sr else '–',
                  f'<span class="badge {r.verdict}">{r.verdict}</span>' if sr else '–',
                  *(f'{getattr(r, k):.3f}' if sr else '–' for k in ('PSNR', 'SSIM', 'LPIPS')), str(int(r.false_positives))]
        body.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
    tbl = f'<table class="t"><tr>{"".join(f"<th>{h}</th>" for h in head)}</tr>{"".join(body)}</table>'
    ge = pd.read_csv(g.ROOT / 'results/geo_eval.csv')
    bufs = sorted(ge.buffer_lr.unique())
    sens = ['<tr><th>method</th>' + ''.join(f'<th>{b} LR px: AP50 / gain / verdict</th>' for b in bufs) + '</tr>']
    for m in ge.method.unique():
        cells = []
        for b in bufs:
            r = ge[(ge.method == m) & (ge.buffer_lr == b)].iloc[0]
            cells.append(f'{r.AP50:.3f}' + (f' / {r.gain:+.3f} / <span class="badge {r.verdict}">{r.verdict}</span>'
                                             if isinstance(r.verdict, str) else ''))
        sens.append(f'<tr><td>{escape(m)}</td>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
    n_gt = ' / '.join(f'{b} px: {int(ge[ge.buffer_lr == b].n_gt.iloc[0])}' for b in bufs)
    rho_txt = ' · '.join(f'{k}: ρ={a.statistic:.2f} (p={a.pvalue:.3f}, n=8), S6 제외 ρ={e.statistic:.2f} (p={e.pvalue:.3f}, n=7)'
                         for k, (a, e) in rho.items())
    html = f'''<h2>요약 (제외 폭 {g.CFG["eval_buffer_lr"]} LR px, 블록 부트스트랩 {g.CFG["bootstrap"]}회 95% CI)</h2>{tbl}
<p class="note">{OFFICIAL}</p>
<h2>그림</h2><div class="charts"><img src="{chart_src['a_forest.png']}"><img src="{chart_src['b_psnr.png']}">
<img src="{chart_src['b_lpips.png']}"><img src="{chart_src['c_recall.png']}"></div><p class="note">Spearman(화질 지표 순위 vs AP50 순위): {rho_txt}</p>
<h2>(d) 제외 폭 민감도 (GT {n_gt})</h2><table class="t">{"".join(sens)}</table>'''
    if POSTHOC:
        html += posthoc_html(chart_src)
    return html


def csv_table(df, fmt=4):
    rows_ = ['<tr>' + ''.join(f'<th>{escape(str(c))}</th>' for c in df.columns) + '</tr>']
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append(f'{v:.{fmt}f}')
            elif isinstance(v, str) and v in ('useful', 'harmful', 'inconclusive', 'better', 'worse'):
                cells.append(f'<span class="badge {v}">{v}</span>')
            else:
                cells.append(escape(str(v)))
        rows_.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
    return f'<div class="scroll"><table class="t">{"".join(rows_)}</table></div>'


def posthoc_html(chart_src):
    fit = json.loads((POST / 'color_fit.json').read_text())
    parts = ['<section class="posthoc"><h2>사후 분석 <span class="badge post">사후 분석, 주 결론 아님</span></h2>',
             '<p class="note">docs/frozen_eval.md 재실행 기록. 탐지기·평가 설정은 고정 값 그대로, 제외 폭 8 LR px.</p>',
             '<h3>S0 대비 이득</h3>', csv_table(pd.read_csv(POST / 'gain_vs_S0.csv')),
             (f'<h3>색 보정 대조군 (밴드별 선형 변환, train MS–PS 쌍: gain {np.round(fit["gain"], 3).tolist()}, '
              f'offset {np.round(fit["offset"], 2).tolist()}, R² {np.round(fit["r2"], 3).tolist()})</h3>'),
             csv_table(pd.read_csv(POST / 'color_control.csv')),
             '<h3>면적 구간별 재현율과 거짓 양성 (운영 conf)</h3>', csv_table(pd.read_csv(POST / 'area_recall_fp.csv'), 3),
             '<h3>Spearman (S6 제외 포함)</h3>', csv_table(pd.read_csv(POST / 'spearman_excl_S6.csv'))]
    for k in sorted(k for k in chart_src if k.startswith('qualitative_')):
        parts.append(f'<img class="wide" src="{chart_src[k]}">')
    return ''.join(parts) + '</section>'


def heatmap_html(data_all, clickable):
    cols = [m for m in METHODS if m not in ('HR', 'LR')]
    d = np.array([[np.nan if t['m'][m]['F1'] is None or t['m']['LR']['F1'] is None else t['m'][m]['F1'] - t['m']['LR']['F1']
                   for m in cols] for t in data_all])
    lim = np.nanpercentile(np.abs(d), 98) or 1
    head = '<tr><th>tile</th><th>GT</th><th>LR F1</th>' + ''.join(f'<th>{short(m)}</th>' for m in cols) + '</tr>'
    rows_ = []
    for t, row in zip(data_all, d):
        cells = []
        for v in row:
            if np.isnan(v):
                cells.append('<td class="na">–</td>')
                continue
            a = min(abs(v) / lim, 1)
            rgb = (int(255 - a * 200), int(255 - a * 130), 255) if v > 0 else (255, int(255 - a * 190), int(255 - a * 190))
            cells.append(f'<td style="background:rgb{rgb}">{v:+.2f}</td>')
        lf = t['m']['LR']['F1']
        cls = ' class="go"' if t['id'] in clickable else ' class="off"'
        rows_.append(f'<tr{cls} data-tile="{t["id"]}"><td>{t["id"]}</td><td>{t["n_gt"]}</td>'
                     f'<td>{"–" if lf is None else f"{lf:.2f}"}</td>{"".join(cells)}</tr>')
    better = np.nansum(d > 0, axis=0)
    worse = np.nansum(d < 0, axis=0)
    summ = '<table class="t"><tr><th>method</th><th>LR보다 나아진 타일</th><th>나빠진 타일</th><th>같음</th><th>비교 가능 타일</th></tr>' + ''.join(
        f'<tr><td>{escape(m)}</td><td>{int(b)}</td><td>{int(w)}</td><td>{int((~np.isnan(d[:, i])).sum() - b - w)}</td>'
        f'<td>{int((~np.isnan(d[:, i])).sum())}</td></tr>' for i, (m, b, w) in enumerate(zip(cols, better, worse))) + '</table>'
    return f'''<h2>타일별 F1 차이 (방법 − LR, 운영 conf, 병합 전 타일 기준)</h2>{summ}
<p class="note">{OFFICIAL} 색: 파랑 = LR보다 개선, 빨강 = 악화(|차이| 98번째 백분위 {lim:.2f}에서 포화). 행을 누르면 타일 뷰어로 이동합니다.</p>
<div class="scroll heat"><table class="t hm">{head}{"".join(rows_)}</table></div>'''


def pick_highlights(data_all):
    """S6 이득 상위 3, S6 손해 상위 2, S4·S7 손해 상위 각 2, S3−S0 상위 1, S6 이득 중앙값 근처 2(GT 10개 이상 타일, 중복 제외)."""
    f1 = pd.DataFrame([{**{m: t['m'][m]['F1'] for m in METHODS}, 'tile_id': t['id'], 'n_gt': t['n_gt']} for t in data_all])
    f1 = f1[f1.n_gt >= 10].set_index('tile_id').astype(float)
    s6, s4 = f1.S6_realesrgan_ft_spacenet - f1.LR, f1.S4_realesrgan_x4plus - f1.LR
    s7, s3 = f1.S7_osediff_x4 - f1.LR, f1.S3_realesrnet_x4plus - f1.S0_gdal_cubic
    rules = [('S6 gain top (F1 S6 − LR)', s6, False, 3), ('S6 loss top (F1 S6 − LR)', s6, True, 2),
             ('S4 loss top (F1 S4 − LR)', s4, True, 2), ('S7 loss top (F1 S7 − LR)', s7, True, 2),
             ('S3 over S0 top (F1 S3 − S0)', s3, False, 1),
             ('near median S6 gain (|F1 S6 − LR − median|)', -(s6 - s6.median()).abs(), False, 2)]
    out, seen = [], set()
    for name, v, asc, k in rules:
        for tid in v.dropna().sort_values(ascending=asc).index:
            if k == 0:
                break
            if tid in seen:
                continue
            seen.add(tid)
            k -= 1
            out.append({'tile_id': tid, 'rule': name, 'value': round(float(v.loc[tid] if 'median' not in name else s6.loc[tid]), 4),
                        'n_gt': int(f1.loc[tid, 'n_gt'])})
    df = pd.DataFrame(out)
    df.to_csv(OUT / 'highlights_tiles.csv', index=False)
    return df


def page(title, overview, data, heat, img_mode, assets, cb_src, extra_note):
    js_assets = json.dumps(assets) if img_mode == 'embed' else 'null'
    return (TEMPLATE.replace('__TITLE__', escape(title)).replace('__OFFICIAL__', OFFICIAL).replace('__OVERVIEW__', overview)
            .replace('__HEAT__', heat).replace('__DATA__', json.dumps(data, separators=(',', ':')))
            .replace('__ASSETS__', js_assets).replace('__CB__', cb_src).replace('__NOTE__', extra_note)
            .replace('__METHODS__', json.dumps([[m, short(m)] for m in METHODS]))
            .replace('__CROPS__', 'true' if (img_mode == 'files' or any(k.endswith('_c00.webp') for k in assets)) else 'false'))


def uri(b, mime):
    return f'data:{mime};base64,' + base64.b64encode(b).decode()


def main():
    tiles = gpd.read_file(g.W / 'tiles.geojson')
    ids = sorted(tiles[(tiles.split == 'test') & tiles.use].tile_id)
    _, union, fp, bld = area('test')
    region, _, _ = gt_in(bld, fp, union, g.CFG['eval_buffer_lr'])
    data, vmax = measure(ids, tiles.set_index('tile_id'), fp, region)
    print('diff vmax (p99 |SR-HR|, S0-S7):', vmax)
    ch, rho = charts()
    for p in sorted(g.FIG.glob('qualitative_*.png')) if POSTHOC else []:
        im = Image.open(p).convert('RGB')
        im.thumbnail((1400, 1400))
        b = io.BytesIO()
        im.save(b, 'JPEG', quality=85)
        ch[p.stem + '.jpg'] = b.getvalue()
    cb = colorbar(vmax)
    # A. 전체
    shutil.rmtree(FULL, ignore_errors=True)
    (FULL / 'images/charts').mkdir(parents=True)
    for k, b in ch.items():
        (FULL / 'images/charts' / k).write_bytes(b)
    (FULL / 'images/colorbar.png').write_bytes(cb)
    with ProcessPoolExecutor(g.CFG['n_proc']) as ex:
        for tid, assets in ex.map(render, [(t, vmax, 85, 85, True, P, region, fp) for t in ids]):
            (FULL / 'images' / tid).mkdir()
            for k, b in assets.items():
                (FULL / 'images' / k).write_bytes(b)
    heat_all = heatmap_html(data, set(ids))
    note = (f'차이 지도 vmax = {vmax:.1f} (S0~S7 전체 test 타일 평가 영역 |SR−HR| RGB 평균의 99번째 백분위). WebP 화질 85, '
            '확대 크롭은 원본 1024의 256 px 격자 크롭을 이어 붙인다.')
    (FULL / 'index.html').write_text(page('SR → 건물 탐지 결과 리포트 (test 96타일)',
                                          overview_html({k: f'images/charts/{k}' for k in ch}, rho), data, heat_all, 'files',
                                          {}, 'images/colorbar.png', note))
    full_mb = sum(f.stat().st_size for f in FULL.rglob('*') if f.is_file()) / 1e6
    # B. 대표 타일 12개: 15 MB 이하가 될 때까지 화질·크롭을 줄인다.
    hl = pick_highlights(data)
    hl_ids = hl.tile_id.tolist()
    hl_data = [t for t in data if t['id'] in hl_ids]
    ov = overview_html({k: uri(b, 'image/png' if k.endswith('.png') else 'image/jpeg') for k, b in ch.items()}, rho)
    heat = heatmap_html(data, set(hl_ids))
    crops = False  # 원본 크롭까지 넣으면 15 MB를 넘는다(85/85에서 50.7 MB, 60/45에서 29.3 MB)
    for q_img, q_map in ((80, 70), (70, 60), (60, 50), (50, 40)):
        with ProcessPoolExecutor(g.CFG['n_proc']) as ex:
            assets = {k: uri(b, 'image/png' if k.endswith('.png') else 'image/webp')
                      for _, a in ex.map(render, [(t, vmax, q_img, q_map, crops, P // 2, region, fp) for t in hl_ids])
                      for k, b in a.items()}
        note = (f'차이 지도 vmax = {vmax:.1f}. 단일 파일 15 MB 제한으로 WebP 화질 원본 {q_img} / 차이 지도 {q_map}, '
                '차이 지도는 256 px(LR 1화소 = HR 4×4 평균)로 저장, 원본 확대 크롭 없음(확대 창은 512 패널을 2배 확대).')
        html = page('SR → 건물 탐지 결과 하이라이트 (대표 타일 12개)', ov, hl_data, heat, 'embed', assets, uri(cb, 'image/png'),
                    note)
        size = len(html.encode())
        print(f'highlights q_img {q_img} q_map {q_map} crops {crops}: {size / 1e6:.2f} MB', flush=True)
        if size <= HL_LIMIT:
            break
    assert size <= HL_LIMIT, size
    (OUT / 'highlights.html').write_text(html)
    info = {'full_index': str((FULL / 'index.html').relative_to(g.ROOT)), 'full_mb': round(full_mb, 1),
            'full_files': sum(1 for f in FULL.rglob('*') if f.is_file()), 'highlights': 'results/report/highlights.html',
            'highlights_mb': round(size / 1e6, 2),
            'highlights_quality': {'img': q_img, 'map': q_map, 'crops': crops, 'diff_px': P // 2},
            'diff_vmax': vmax, 'tiles': len(ids), 'methods': METHODS}
    (OUT / 'report_viewer.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    print(hl.to_string())


TEMPLATE = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{--bg:#fafaf8;--ink:#1d1d1b;--mut:#666;--line:#ddd;--card:#fff;--acc:#2b6cb0}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--ink:#eee;--mut:#aaa;--line:#333;--card:#1f1f1f;--acc:#7fb2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5}
h1{font-size:18px;margin:0 0 4px}h2{font-size:16px;margin:22px 0 8px}h3{font-size:14px;margin:16px 0 6px}
.official{color:#b45309;font-weight:600;font-size:13px}.note{color:var(--mut);font-size:12.5px}
nav button{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:6px 12px;margin-right:4px;border-radius:6px;cursor:pointer}
nav button.on{background:var(--acc);color:#fff;border-color:var(--acc)}main{padding:0 16px 40px}.tab{display:none}.tab.on{display:block}
table.t{border-collapse:collapse;font-size:12.5px;margin:6px 0}.t th,.t td{border:1px solid var(--line);padding:3px 7px;text-align:right;white-space:nowrap}
.t th{background:var(--card);position:sticky;top:0}.t td:first-child,.t th:first-child{text-align:left}.ci{color:var(--mut);font-size:11.5px}
.badge{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11.5px;color:#fff}.useful,.better{background:#1f9d55}.harmful,.worse{background:#d64545}
.inconclusive{background:#8a8a8a}.post{background:#b45309}.posthoc{border:2px dashed #b45309;padding:4px 12px 12px;border-radius:8px;margin-top:20px}
.charts{display:flex;flex-wrap:wrap;gap:10px}.charts img{max-width:100%;border:1px solid var(--line);background:#fff}img.wide{max-width:100%;margin:8px 0;background:#fff}
.scroll{overflow-x:auto;max-width:100%}.heat{max-height:70vh;overflow:auto}.hm td{color:#111}.hm tr.go{cursor:pointer}.hm tr.go:hover td{outline:1px solid var(--acc)}
.hm tr.off td:first-child{color:var(--mut)}.hm td.na{color:#999}
.ctl{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:10px 0}.ctl label{font-size:13px}
select,input[type=range]{font:inherit}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px}.panel h4{margin:2px 2px 0;font-size:13px}
.panel .mt{font-size:11px;color:var(--mut);margin:0 2px 3px;min-height:28px}.stack{position:relative;width:100%;aspect-ratio:1;cursor:crosshair}
.stack img,.stack svg,.stack canvas,.stack .blk{position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated}
.stack .blk{background:#000}.stack.iso{isolation:isolate}.stack img.scr{mix-blend-mode:screen}
.zoomgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px;margin-top:10px}.zoomgrid canvas{width:100%;image-rendering:pixelated;border:1px solid var(--line)}
.zoomgrid div{font-size:12px}.cmp{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}.slider{position:relative;width:min(512px,100%);aspect-ratio:1}
.slider img{position:absolute;inset:0;width:100%;height:100%}.slider .bar{position:absolute;top:0;bottom:0;width:2px;background:#ffd400}
.ref{width:min(512px,100%)}.ref img{width:100%}.legend span{display:inline-block;margin-right:12px;font-size:12px}
.sw{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:4px}
</style></head><body><header><h1>__TITLE__</h1><div class="official">__OFFICIAL__</div>
<nav><button data-t="ov" class="on">1. 개요</button><button data-t="tv">2. 타일 뷰어</button><button data-t="hm">3. 타일별 성능 차이</button></nav></header>
<main><div id="ov" class="tab on">__OVERVIEW__</div>
<div id="tv" class="tab"><div class="ctl"><label>정렬 <select id="sort"><option value="id">타일 ID</option><option value="s6">S6 이득 큰 순</option>
<option value="s4">S4 손해 큰 순</option><option value="s7">S7 손해 큰 순</option><option value="ngt">건물 수</option></select></label>
<label>타일 <select id="tile"></select></label><span id="tinfo" class="note"></span></div>
<div class="ctl" id="modes"><label><input type="radio" name="mode" value="img" checked> 원본 영상</label><label><input type="radio" name="mode" value="det"> 탐지 겹치기</label>
<label><input type="radio" name="mode" value="hr"> |SR−HR| 차이</label><label><input type="radio" name="mode" value="s0"> |SR−S0| 차이</label>
<label><input type="radio" name="mode" value="canny"> Canny 엣지</label><label><input type="checkbox" id="maskon" checked> 무효(빗금)·평가 제외(반투명) 표시</label></div>
<div class="ctl legend" id="leg"></div><p class="note">__OFFICIAL__ __NOTE__ 패널을 누르면 모든 방법의 같은 위치 256 px(원본 해상도) 크롭을 아래에 보여 줍니다.</p>
<div class="grid" id="panels"></div><h3 id="zt"></h3><div class="zoomgrid" id="zoom"></div>
<h2>비교 슬라이더 (HR 기준)</h2><div class="ctl"><label>왼쪽 <select id="cl"></select></label><label>오른쪽 <select id="cr"></select></label>
<input type="range" id="cs" min="0" max="100" value="50"></div><div class="cmp"><div class="slider" id="sl"><img id="sr_r"><img id="sr_l"><div class="bar" id="sbar"></div></div>
<div class="ref"><div class="note">HR (기준)</div><img id="sref"></div></div></div>
<div id="hm" class="tab">__HEAT__</div></main>
<script>
const DATA=__DATA__,ASSETS=__ASSETS__,METHODS=__METHODS__,CROPS=__CROPS__,CB="__CB__",N=1024,C=256;
const src=k=>ASSETS?ASSETS[k]:'images/'+k;
const T={};DATA.forEach(t=>T[t.id]=t);
const f=(v,d=2)=>v==null?'–':v.toFixed(d);
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>show(b.dataset.t));
function show(t){document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.t==t));
 document.querySelectorAll('.tab').forEach(d=>d.classList.toggle('on',d.id==t));}
const sel=document.getElementById('tile'),sortSel=document.getElementById('sort');
const d1=(t,m,r)=>{const a=t.m[m].F1,b=t.m[r].F1;return a==null||b==null?null:a-b};
function order(){const k=sortSel.value,ids=DATA.map(t=>t.id);
 const kf={id:t=>0,s6:t=>-(d1(t,'S6_realesrgan_ft_spacenet','LR')??-9),s4:t=>d1(t,'S4_realesrgan_x4plus','LR')??9,
  s7:t=>d1(t,'S7_osediff_x4','LR')??9,ngt:t=>-t.n_gt}[k];
 ids.sort((a,b)=>kf(T[a])-kf(T[b])||(a<b?-1:1));const cur=sel.value;
 sel.innerHTML=ids.map(i=>{const t=T[i],g=d1(t,'S6_realesrgan_ft_spacenet','LR');return `<option value="${i}">${i} (GT ${t.n_gt}, S6−LR F1 ${f(g)})</option>`}).join('');
 sel.value=cur&&T[cur]?cur:ids[0];draw();}
sortSel.onchange=order;sel.onchange=draw;
document.querySelectorAll('input[name=mode]').forEach(r=>r.onchange=draw);document.getElementById('maskon').onchange=draw;
const mode=()=>document.querySelector('input[name=mode]:checked').value;
function key(t,m,md){if(md=='hr'&&m=='HR')return null;if(md=='s0'&&m=='S0_gdal_cubic')return null;
 return `${t}/${m}_${md=='det'?'img':md}.webp`}
function layers(t,m,md){if(md!='canny'){const k=key(t,m,md);return k?`<img src="${src(k)}">`:'<div class="note" style="padding:40%">기준 (차이 없음)</div>'}
 return `<div class="blk"></div><img src="${src(t+'/edge_HR.png')}">`+(m=='HR'?'':`<img class="scr" src="${src(t+'/'+m+'_edge.png')}">`)}
function svg(t,m){const e=T[t],r=e.m[m],p=a=>a.slice(0,8).join(' ').replace(/(\S+) (\S+)/g,'$1,$2');let s='';
 e.gt.forEach(g=>s+=`<polygon points="${p(g)}" fill="none" stroke="#ffd400" stroke-width="3"/>`);
 r.fn.forEach(i=>s+=`<polygon points="${p(e.gt[i])}" fill="none" stroke="#2f7bff" stroke-width="4" stroke-dasharray="10 7"/>`);
 r.det.forEach(d=>s+=`<polygon points="${p(d)}" fill="none" stroke="${d[8]?'#19c37d':'#ff3b30'}" stroke-width="3"/>`);
 return `<svg viewBox="0 0 ${N} ${N}">${s}</svg>`}
function legend(){const md=mode(),L=document.getElementById('leg');
 if(md=='det')L.innerHTML='<span><i class="sw" style="background:#ffd400"></i>GT</span><span><i class="sw" style="background:#19c37d"></i>TP</span><span><i class="sw" style="background:#ff3b30"></i>FP</span><span><i class="sw" style="background:#2f7bff"></i>FN (점선)</span>';
 else if(md=='hr'||md=='s0')L.innerHTML=`<img src="${CB}" style="height:48px;background:#fff">`;
 else if(md=='canny')L.innerHTML='<span><i class="sw" style="background:#28c846"></i>HR 엣지</span><span><i class="sw" style="background:#eb3c32"></i>SR 엣지</span><span><i class="sw" style="background:#fadc28"></i>겹침</span><span class="note">Canny 47/94(frozen), 512 패널은 2×2 최대값 축소</span>';
 else L.innerHTML='';}
function draw(){const t=sel.value,e=T[t],md=mode(),mk=document.getElementById('maskon').checked;legend();
 document.getElementById('tinfo').textContent=`GT ${e.n_gt}, 유효 비율 ${e.valid_frac}`;
 document.getElementById('panels').innerHTML=METHODS.map(([m,s])=>{const r=e.m[m],k=key(t,m,md);
  const mt=`PSNR ${f(r.psnr)} · LPIPS ${f(r.lpips,3)}<br>P ${f(r.P)} R ${f(r.R)} F1 ${f(r.F1)} · TP ${r.TP} FP ${r.FP} FN ${r.FN}`;
  return `<div class="panel"><h4>${s}</h4><div class="mt">${mt}</div><div class="stack${md=='canny'?' iso':''}" data-m="${m}">`+
   layers(t,m,md)+
   (mk?`<img src="${src(t+'/mask.png')}">`:'')+(md=='det'?svg(t,m):'')+'</div></div>'}).join('');
 document.querySelectorAll('.stack').forEach(s=>s.onclick=ev=>{const b=s.getBoundingClientRect();
  zoom((ev.clientX-b.left)/b.width*N,(ev.clientY-b.top)/b.height*N)});
 document.getElementById('zoom').innerHTML='';document.getElementById('zt').textContent='';slider();}
const load=u=>new Promise(r=>{const i=new Image();i.onload=()=>r(i);i.onerror=()=>r(null);i.src=u});
async function zoom(cx,cy){const t=sel.value,e=T[t],md=mode(),mk=document.getElementById('maskon').checked;
 const x0=Math.round(Math.min(Math.max(cx-C/2,0),N-C)),y0=Math.round(Math.min(Math.max(cy-C/2,0),N-C));
 document.getElementById('zt').textContent=`확대: x ${x0}–${x0+C}, y ${y0}–${y0+C} (HR px, 원본 해상도)`;
 const Z=document.getElementById('zoom');Z.innerHTML='';const mask=mk?await load(src(t+'/mask.png')):null;
 for(const [m,s] of METHODS){const cv=document.createElement('canvas');cv.width=C;cv.height=C;const x=cv.getContext('2d');x.imageSmoothingEnabled=false;
  const k=key(t,m,md);
  if(k&&(md=='img'||md=='det')&&CROPS){for(let i=Math.floor(y0/C);i<=Math.floor((y0+C-1)/C);i++)for(let j=Math.floor(x0/C);j<=Math.floor((x0+C-1)/C);j++){
    const im=await load(src(`${t}/${m}_c${i}${j}.webp`));if(im)x.drawImage(im,j*C-x0,i*C-y0);}}
  else if(md=='canny'){x.fillStyle='#000';x.fillRect(0,0,C,C);const ls=[t+'/edge_HR.png'].concat(m=='HR'?[]:[t+'/'+m+'_edge.png']);
   for(const u of ls){const im=await load(src(u));if(im){const s=im.naturalWidth/N;x.globalCompositeOperation='screen';x.drawImage(im,x0*s,y0*s,C*s,C*s,0,0,C,C);}}
   x.globalCompositeOperation='source-over';}
  else if(k){const im=await load(src(k));if(im){const s=im.naturalWidth/N;x.drawImage(im,x0*s,y0*s,C*s,C*s,0,0,C,C);}}
  if(mask)x.drawImage(mask,x0,y0,C,C,0,0,C,C);
  if(md=='det'){const r=e.m[m],poly=(a,c,w,dash)=>{x.beginPath();for(let q=0;q<4;q++)x[q?'lineTo':'moveTo'](a[2*q]-x0,a[2*q+1]-y0);
    x.closePath();x.strokeStyle=c;x.lineWidth=w;x.setLineDash(dash||[]);x.stroke()};
   e.gt.forEach(g=>poly(g,'#ffd400',1.5));r.fn.forEach(i=>poly(e.gt[i],'#2f7bff',2,[5,4]));r.det.forEach(d=>poly(d,d[8]?'#19c37d':'#ff3b30',1.5));}
  const w=document.createElement('div');w.innerHTML=`<div>${s}</div>`;w.appendChild(cv);Z.appendChild(w);}}
const cl=document.getElementById('cl'),cr=document.getElementById('cr'),cs=document.getElementById('cs');
cl.innerHTML=cr.innerHTML=METHODS.map(([m,s])=>`<option value="${m}">${s}</option>`).join('');cl.value='LR';cr.value='S6_realesrgan_ft_spacenet';
function slider(){const t=sel.value;document.getElementById('sr_l').src=src(`${t}/${cl.value}_img.webp`);
 document.getElementById('sr_r').src=src(`${t}/${cr.value}_img.webp`);document.getElementById('sref').src=src(`${t}/HR_img.webp`);cut();}
function cut(){const v=cs.value;document.getElementById('sr_l').style.clipPath=`inset(0 ${100-v}% 0 0)`;document.getElementById('sbar').style.left=v+'%';}
cl.onchange=cr.onchange=slider;cs.oninput=cut;
document.querySelectorAll('.hm tr.go').forEach(r=>r.onclick=()=>{sortSel.value='id';order();sel.value=r.dataset.tile;draw();show('tv');window.scrollTo(0,0)});
order();
</script></body></html>'''


if __name__ == '__main__':
    main()
