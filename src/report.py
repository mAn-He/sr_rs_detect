"""results/summary.csv(행: HR 상한, 경로 1, 경로 2 SR 방법), 화질 지표 순위와 AP50 순위의 Spearman 상관, 그림 (a)~(h).
`readme` 인자: README.md·README.ko.md의 결과표를 results/ CSV와 logs/timing.csv에서 다시 쓰고,
두 파일의 모든 수치·표·그림 경로·상대 링크를 대조해 docs/code_check.md에 남긴다."""
import json
import re
import sys
from collections import Counter

import cv2
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import spearmanr

import gdal_ops as g
from det_infer import rows
from geo_eval import BINS, EDGES, inside, match, obb

matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = g.ROOT / 'results'
INK, MUTED, SERIES = '#0b0b0b', '#52514e', ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
# 산점도 라벨 오프셋(점): S5·S7은 LPIPS·AP50(그리고 처리 시간)이 거의 같아 S5 라벨을 점 아래에 둔다.
LABEL_XY = {'S5': (6, -12)}
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})


def summary():
    ev = pd.read_csv(R / 'geo_eval.csv')
    main = ev[ev.buffer_lr == g.CFG['eval_buffer_lr']].set_index('method')
    srm = pd.read_csv(R / 'sr_metrics.csv').set_index('method')
    srt = pd.read_csv(g.W / 'sr_timing.csv').set_index('method')
    det = pd.read_csv(g.W / 'det_timing.csv').set_index('method')
    out = []
    for name, *_ in rows('test'):
        cat = {'HR': 'upper_bound', 'LR': 'lower_bound'}.get(name, g.CFG['sr_methods'].get(name))
        path = {'HR': 'HR (D_HR on HR)', 'LR': 'path1 (D_LR on LR)'}.get(name, 'path2 (D_HR on SR)')
        r = {'method': name, 'category': cat, 'path': path}
        for k in ('PSNR', 'SSIM', 'LPIPS', 'edge_precision', 'edge_recall', 'sharpness_ratio'):
            r[k] = srm.at[name, k] if name in srm.index else np.nan
        e = main.loc[name]
        r.update({'AP50_geo': e.AP50, 'AP50_ci_low': e.AP50_ci_low, 'AP50_ci_high': e.AP50_ci_high})
        for b in BINS:
            r.update({f'recall_{b}': e[f'recall_{b}'], f'recall_{b}_ci_low': e[f'recall_{b}_ci_low'],
                      f'recall_{b}_ci_high': e[f'recall_{b}_ci_high']})
        for k in ('gain', 'gain_ci_low', 'gain_ci_high', 'recovery', 'verdict'):
            r[k] = e.get(k, np.nan)
        r['false_positives'] = e.false_positives
        # 처리 시간은 모델 로딩·예열을 뺀 값, 로딩·예열 시간은 별도 열(보간 S0·S1은 모델이 없어 비어 있다)
        r['sr_sec_per_km2'] = srt.at[name, 'seconds'] / srt.at[name, 'km2'] if name in srt.index else np.nan
        r['det_sec_per_km2'] = det.at[name, 'seconds'] / det.at[name, 'km2']
        r['sr_load_sec'] = srt.at[name, 'load_sec'] if name in srt.index else np.nan
        r['det_load_sec'] = det.at[name, 'load_sec']
        r['conf_op'] = e.conf_op
        for b in BINS:
            for k in (f'gain_recall_{b}', f'gain_recall_{b}_ci_low', f'gain_recall_{b}_ci_high'):
                r[k] = e.get(k, np.nan)
            r[f'verdict_recall_{b}'] = e.get(f'verdict_recall_{b}', np.nan)
        for buf in g.CFG['eval_buffer_sens']:
            s = ev[(ev.buffer_lr == buf) & (ev.method == name)].iloc[0]
            r.update({f'AP50_geo_buf{buf}': s.AP50, f'gain_buf{buf}': s.get('gain', np.nan),
                      f'verdict_buf{buf}': s.get('verdict', np.nan)})
        out.append(r)
    df = pd.DataFrame(out)
    df.to_csv(R / 'summary.csv', index=False)
    print(df[['method', 'PSNR', 'SSIM', 'LPIPS', 'AP50_geo', 'gain', 'verdict', 'false_positives']].to_string())
    return df


def spearman(df):
    sr = df[df.path.str.startswith('path2')]
    rows_ = []
    for k, sign in (('PSNR', 1), ('SSIM', 1), ('LPIPS', -1), ('edge_precision', 1), ('edge_recall', 1)):
        rho, p = spearmanr(sign * sr[k], sr.AP50_geo)
        rows_.append({'metric': k, 'higher_is_better': sign == 1, 'spearman_rho_vs_AP50': rho, 'p_value': p,
                      'n_methods': len(sr)})
    s = pd.DataFrame(rows_)
    s.to_csv(R / 'spearman.csv', index=False)
    print(s.to_string())


def scatter(df, x, fname, xlabel, logx=False):
    sr = df[df.path.str.startswith('path2')]
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    for name, ls in (('HR', '--'), ('LR', ':')):
        ax.axhline(df.set_index('method').at[name, 'AP50_geo'], color=MUTED, ls=ls, lw=1)
        ax.text(1.0, df.set_index('method').at[name, 'AP50_geo'], f' {name}', transform=ax.get_yaxis_transform(),
                color=MUTED, va='center')
    ax.errorbar(sr[x], sr.AP50_geo, yerr=[sr.AP50_geo - sr.AP50_ci_low, sr.AP50_ci_high - sr.AP50_geo], fmt='o',
                ms=8, color=SERIES[0], ecolor=SERIES[0], elinewidth=1.5, capsize=0)
    for r in sr.itertuples():
        m = r.method.split('_')[0]
        ax.annotate(m, (getattr(r, x), r.AP50_geo), xytext=LABEL_XY.get(m, (6, 4)), textcoords='offset points',
                    color=INK)
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('AP50_geo (95% CI, test-block bootstrap)')
    ax.grid(axis='y', color='#e6e5e0', lw=0.6)
    fig.savefig(g.FIG / fname, dpi=130)
    plt.close(fig)


def recall_bars(df):
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(df)), 4.2), constrained_layout=True)
    w = 0.8 / len(BINS)
    x = np.arange(len(df))
    for k, b in enumerate(BINS):
        v = df[f'recall_{b}']
        ax.bar(x + (k - (len(BINS) - 1) / 2) * w, v, w * 0.92, color=SERIES[k], label=f'{b} m²',
               yerr=[v - df[f'recall_{b}_ci_low'], df[f'recall_{b}_ci_high'] - v], ecolor=MUTED, error_kw={'lw': 0.8})
    ax.set_xticks(x, [m.split('_')[0] for m in df.method])
    c = df.set_index('method').conf_op
    ax.set_ylabel(f"recall at operating conf (95% CI)\nD_HR {c['HR']:.3f}, D_LR {c['LR']:.3f} (val F1 max)")
    ax.legend(frameon=False, ncol=len(BINS), loc='upper left', bbox_to_anchor=(0, 1.12))
    ax.grid(axis='y', color='#e6e5e0', lw=0.6)
    fig.savefig(g.FIG / 'report_b_recall_by_area.png', dpi=130)
    plt.close(fig)


def read_crop(path, geom):
    with rasterio.open(path) as s:
        return s.read(window=g.window(s, geom)).transpose(1, 2, 0)


def scenes(df):
    """(c) 같은 장면의 방법별 비교 그리드, (d) 운영점 거짓 양성이 가장 많은 SR 방법의 최고 conf 거짓 양성 확대
    (conf 제한 없이 고르고 conf를 제목에 적는다)."""
    tiles = gpd.read_file(g.W / 'tiles.geojson')
    test = tiles[(tiles.split == 'test') & tiles.use]
    lab = pd.read_csv(g.W / 'labels_buildings.csv')
    t = test.set_index('tile_id').loc[lab[lab.split == 'test'].tile_id.value_counts().index[0]]
    x0, y0, x1, y1 = t.geometry.bounds
    cx, cy = g.LR * round((x0 + x1) / 2 / g.LR), g.LR * round((y0 + y1) / 2 / g.LR)
    geom = gpd.GeoSeries.from_xy([cx], [cy]).buffer(38.4, cap_style='square').iloc[0]  # 76.8 m = HR 256 px
    names = [r[0] for r in rows('test')]
    imgs = []
    for n in names:
        kind = {'HR': 'tiles/hr', 'LR': 'tiles/lr'}.get(n, f'sr/{n}')
        im = read_crop(g.W / kind / f'images/test/{t.name}.tif', geom)
        imgs.append(cv2.resize(im, (256, 256), interpolation=cv2.INTER_NEAREST))
    cols = 5
    fig, axs = plt.subplots(int(np.ceil(len(imgs) / cols)), cols, figsize=(2.4 * cols, 2.6 * np.ceil(len(imgs) / cols)),
                            constrained_layout=True)
    for ax in axs.ravel():
        ax.axis('off')
    for ax, n, im in zip(axs.ravel(), names, imgs):
        ax.imshow(im)
        ax.set_title(n.split('_')[0] if n not in ('HR', 'LR') else n, color=INK)
    fig.suptitle(f'tile {t.name}, 76.8 m crop (HR 256 px)', color=MUTED)
    fig.savefig(g.FIG / 'report_c_method_grid.png', dpi=130)
    plt.close(fig)
    # (d)
    sr = df[df.path.str.startswith('path2')]
    m = sr.sort_values('false_positives').method.iloc[-1]
    fp_ = gpd.read_file(g.W / 'footprint.geojson').geometry.iloc[0]
    region = fp_.buffer(-g.CFG['eval_buffer_lr'] * g.LR).intersection(test.union_all())
    bld = gpd.read_file(g.W / 'buildings_utm.geojson')
    gt = bld[bld.intersects(region)]
    gt = gt[inside(gt.geometry, region)]
    gt_obb = np.array([obb(x) for x in gt.geometry])
    p = gpd.read_file(g.W / 'preds/test' / f'{m}.geojson').to_crs(g.UTM)
    p = p[inside(p.geometry, region)].reset_index(drop=True)
    tp, _ = match(p, gt_obb)
    fps = p[~tp].sort_values('conf', ascending=False)
    assert len(fps), f'{m}: no false positives'
    f = fps.iloc[0]
    c = f.geometry.centroid
    tid = f.tile_id
    tg = tiles.set_index('tile_id').loc[tid].geometry
    bx0, by0, bx1, by1 = tg.bounds
    cx = min(max(g.LR * round(c.x / g.LR), bx0 + 19.2), bx1 - 19.2)
    cy = min(max(g.LR * round(c.y / g.LR), by0 + 19.2), by1 - 19.2)
    geom = gpd.GeoSeries.from_xy([cx], [cy]).buffer(19.2, cap_style='square').iloc[0]  # 38.4 m = HR 128 px
    fig, axs = plt.subplots(1, 2, figsize=(8, 4.3), constrained_layout=True)
    for ax, kind, title in ((axs[0], f'sr/{m}', m), (axs[1], 'tiles/hr', 'HR')):
        ax.imshow(read_crop(g.W / kind / f'images/test/{tid}.tif', geom), extent=(cx - 19.2, cx + 19.2, cy - 19.2, cy + 19.2))
        for o in gt_obb:
            if o.intersects(geom):
                ax.plot(*o.exterior.xy, color='#1baf7a', lw=1.5)
        ax.plot(*f.geometry.exterior.xy, color='#e34948', lw=2)
        ax.set_xlim(cx - 19.2, cx + 19.2)
        ax.set_ylim(cy - 19.2, cy + 19.2)
        ax.axis('off')
        ax.set_title(title, color=INK)
    fig.suptitle(f'(d) highest-confidence false positive of {m} (conf {f.conf:.2f}, red) vs GT OBB (green)', color=MUTED)
    fig.savefig(g.FIG / 'report_d_false_positive.png', dpi=130)
    plt.close(fig)


def forest(df):
    """(g) gain 포리스트 플롯: 판정을 색과 글자로 함께 표시한다."""
    sr = df[df.path.str.startswith('path2')].iloc[::-1]
    col = {'useful': '#1baf7a', 'harmful': '#e34948', 'inconclusive': MUTED}
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    y = np.arange(len(sr))
    for yi, r in zip(y, sr.itertuples()):
        ax.errorbar(r.gain, yi, xerr=[[r.gain - r.gain_ci_low], [r.gain_ci_high - r.gain]], fmt='o', ms=7,
                    color=col[r.verdict], elinewidth=2, capsize=0)
        ax.annotate(r.verdict, (r.gain_ci_high, yi), xytext=(6, 0), textcoords='offset points', va='center',
                    color=MUTED, fontsize=8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, [m.split('_')[0] for m in sr.method])
    ax.margins(x=0.18)
    ax.set_xlabel('gain = AP50_geo(D_HR on SR) − AP50_geo(D_LR on LR)\n95% CI, test-block bootstrap')
    ax.grid(axis='x', color='#e6e5e0', lw=0.6)
    fig.savefig(g.FIG / 'report_g_gain_forest.png', dpi=130)
    plt.close(fig)


def main():
    df = summary()
    spearman(df)
    scatter(df, 'PSNR', 'report_a_psnr_vs_ap50.png', 'PSNR (dB, evaluation mask)')
    recall_bars(df)
    scenes(df)
    scatter(df, 'sr_sec_per_km2', 'report_e_time_vs_ap50.png', 'SR time (s per km², excl. model load)', logx=True)
    scatter(df, 'sharpness_ratio', 'report_f_sharpness_vs_ap50.png', 'sharpness_ratio (Laplacian var SR/HR)')
    forest(df)
    scatter(df, 'LPIPS', 'report_h_lpips_vs_ap50.png', 'LPIPS (evaluation mask, lower is better)')


# ---- README 결과표와 대조 (readme) ----
DOCS = [g.ROOT / 'README.md', g.ROOT / 'README.ko.md']
BIN_TXT = {b: f'<{EDGES[1]:g}' if i == 0 else f'>{EDGES[i]:g}' if np.isinf(EDGES[i + 1]) else f'{EDGES[i]:g}–{EDGES[i + 1]:g}'
           for i, b in enumerate(BINS)}
PH = R / 'posthoc'


def short(m):
    return m if m in ('HR', 'LR', 'S0_color', 'S3_color') else m.split('_')[0]


def ci(v, lo, hi, sign=False):
    f = '{:+.3f}' if sign else '{:.3f}'
    return f'{f.format(v)} [{f.format(lo)}, {f.format(hi)}]'


def md(head, rows_):
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
                     + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows_])


def stage_times():
    """logs/timing.csv에서 현재 파이프라인 단계의 마지막 실행 시간(단계 합)."""
    t = pd.read_csv(g.ROOT / 'logs/timing.csv')
    t = t[~t.smoke.astype(bool)].drop_duplicates(['stage', 'step'], keep='last')
    t = t[(t.stage != 'det_sens') & ~((t.stage == 'train_det') & ~t.step.str.contains('det_train'))]
    order = ['data', 'raster', 'tiles_labels', 'train_det', 'det_calib', 'sr', 'eval', 'posthoc', 'report_viewer']
    st = t.groupby('stage').agg(seconds=('seconds', 'sum'), gpu=('gpu_util_mean', 'max'),
                                ram=('ram_max_gb', 'max')).reindex(order)
    assert st.seconds.notna().all(), st
    return st


def tables():
    s = pd.read_csv(R / 'summary.csv')
    sr = s.path.str.startswith('path2')
    T = {}
    T['main'] = md(['method', 'path', 'AP50_geo [95% CI]', 'gain vs LR [95% CI]', 'recovery', 'verdict', 'FP'],
                   [[r.method, r.path, ci(r.AP50_geo, r.AP50_ci_low, r.AP50_ci_high),
                     ci(r.gain, r.gain_ci_low, r.gain_ci_high, True) if q else '–', f'{r.recovery:.2f}' if q else '–',
                     f'**{r.verdict}**' if q else '–', int(r.false_positives)] for r, q in zip(s.itertuples(), sr)])
    T['quality'] = md(['method', 'PSNR', 'SSIM', 'LPIPS', 'edge_precision', 'edge_recall', 'sharpness_ratio',
                       'SR s/km²', 'SR load s', 'det s/km²'],
                      [[r.method] + [f'{getattr(r, k):.2f}' if k == 'PSNR' else f'{getattr(r, k):.3f}' for k in
                                     ('PSNR', 'SSIM', 'LPIPS', 'edge_precision', 'edge_recall', 'sharpness_ratio')]
                       + [f'{r.sr_sec_per_km2:.1f}', '–' if np.isnan(r.sr_load_sec) else f'{r.sr_load_sec:.1f}',
                          f'{r.det_sec_per_km2:.1f}'] for r in s[sr].itertuples()])
    T['recall'] = md(['method'] + [f'{BIN_TXT[b]} m²' for b in BINS] + ['verdict vs LR (' + ' / '.join(BIN_TXT[b] for b in BINS) + ')'],
                     [[r.method] + [ci(getattr(r, f'recall_{b}'), getattr(r, f'recall_{b}_ci_low'),
                                       getattr(r, f'recall_{b}_ci_high')) for b in BINS]
                      + [' / '.join(getattr(r, f'verdict_recall_{b}') for b in BINS) if q else '–']
                      for r, q in zip(s.itertuples(), sr)])
    bufs = g.CFG['eval_buffer_sens']
    T['buffer'] = md(['method'] + [f'{b} LR px: AP50 / gain / verdict' for b in bufs],
                     [[r.method] + [f'{getattr(r, f"AP50_geo_buf{b}"):.3f}' + (
                         f' / {getattr(r, f"gain_buf{b}"):+.3f} / {getattr(r, f"verdict_buf{b}")}' if q else '')
                         for b in bufs] for r, q in zip(s.itertuples(), sr)])
    sp = pd.read_csv(R / 'spearman.csv')
    T['spearman'] = md(['metric', 'Spearman ρ vs AP50_geo', 'p', 'n'],
                       [[r.metric, f'{r.spearman_rho_vs_AP50:.3f}', f'{r.p_value:.3f}', r.n_methods] for r in sp.itertuples()])
    t1 = pd.read_csv(PH / 'gain_vs_S0.csv')
    T['ph_s0'] = md(['method', 'AP50_geo [95% CI]', 'AP50 − S0 [95% CI]', 'verdict'],
                    [[r.method, ci(r.AP50_geo, r.AP50_ci_low, r.AP50_ci_high),
                      ci(r.AP50_minus_S0_gdal_cubic, r.diff_ci_low, r.diff_ci_high, True), r.verdict] for r in t1.itertuples()])
    cf = json.loads((PH / 'color_fit.json').read_text())
    T['ph_colorfit'] = md(['band', 'gain', 'offset', 'R²'],
                          [[b, f'{cf["gain"][i]:.3f}', f'{cf["offset"][i]:.2f}', f'{cf["r2"][i]:.3f}']
                           for i, b in enumerate(cf['bands'])])
    t2 = pd.read_csv(PH / 'color_control.csv')
    T['ph_color'] = md(['method', 'AP50_geo [95% CI]', 'reference', 'reference AP50', 'difference [95% CI]', 'verdict'],
                       [[r.method, ci(r.AP50_geo, r.AP50_ci_low, r.AP50_ci_high), r.reference, f'{r.reference_AP50:.3f}',
                         ci(r.AP50_minus_ref, r.diff_ci_low, r.diff_ci_high, True), r.verdict] for r in t2.itertuples()])
    t3 = pd.read_csv(PH / 'area_recall_fp.csv')
    T['ph_area'] = md(['method', 'conf'] + [f'{BIN_TXT[b]} m²' for b in BINS] + ['FP', 'predictions', 'FP ratio'],
                      [[r.method, f'{r.conf_op:.3f}'] + [ci(getattr(r, f'recall_{b}'), getattr(r, f'recall_{b}_ci_low'),
                                                           getattr(r, f'recall_{b}_ci_high')) for b in BINS]
                       + [r.false_positives, r.n_pred_op, f'{r.fp_ratio:.3f}'] for r in t3.itertuples()])
    t4 = pd.read_csv(PH / 'spearman_excl_S6.csv')
    T['ph_spearman'] = md(['metric', 'set', 'ρ', 'p'],
                          [[r.metric, r.set, f'{r.spearman_rho_vs_AP50:.3f}', f'{r.p_value:.3f}'] for r in t4.itertuples()])
    st = stage_times()
    T['timing'] = md(['stage', 'minutes', 'max mean GPU util %', 'max RAM GB'],
                     [[k, f'{r.seconds / 60:.1f}', f'{r.gpu:.0f}', f'{r.ram:.1f}'] for k, r in st.iterrows()]
                     + [['total', f'{st.seconds.sum() / 60:.1f} ({st.seconds.sum() / 3600:.2f} h)', '', '']])
    derived = [*(st.seconds / 60), st.seconds.sum() / 60, st.seconds.sum() / 3600]
    return T, derived


NUM = re.compile(r'(?<![\w.\-−+])[-−+]?\d[\d,]*(?:\.\d+)?(?![\w.]*\d)')


def numbers(text):
    """표 밖 코드·URL·링크 대상·HTML 주석을 빼고 수치 토큰을 뽑는다."""
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    text = re.sub(r'<!--.*?-->|`[^`]*`|\]\([^)]*\)|https?://\S+', ' ', text)
    return [(m.group(), text[m.end():m.end() + 1] == '%') for m in NUM.finditer(text)]


def norm(tok):
    return tok.replace(',', '').replace('−', '-').lstrip('+')


def sources(derived):
    vals = list(derived)
    files = [*(p for p in R.rglob('*.csv') if 'report/full' not in str(p)), g.ROOT / 'logs/timing.csv']
    for p in files:
        for c in pd.read_csv(p).select_dtypes('number').to_numpy().ravel():
            vals.append(float(c))

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            vals.append(float(o))
    for p in R.rglob('*.json'):
        walk(json.loads(p.read_text()))
    vals = np.array([v for v in vals if np.isfinite(v)])
    num = {d: set(np.round(vals, d)) | set(np.round(np.abs(vals), d)) for d in range(5)}
    txt = set()
    for p in [*sorted((g.ROOT / 'docs').glob('*.md')), g.ROOT / 'configs/default.yaml', g.ROOT / 'third_party/VERSIONS.md']:
        if p.name != 'code_check.md':
            txt |= {norm(t).lstrip('-') for t, _ in numbers(p.read_text())}
    return num, txt


# 결과 파일에 없는 수치(인용 연도, 저장소 고정 정보). 이유를 함께 둔다.
ALLOW = {'2019': 'Shermeyer & Van Etten 2019 인용 연도', '1812.04098': 'Shermeyer & Van Etten arXiv 번호',
         '2605.00310': 'GeoSR-Bench arXiv 번호(CLAUDE.md)', '4.0': 'CC BY-SA 4.0 라이선스 버전',
         '2.0': 'Apache-2.0 라이선스 버전', '1.0': 'S-Lab License 1.0 버전', '3': 'AGPL-3.0·BSD-3-Clause 버전'}


def classify(tok, pct, num, txt):
    n = norm(tok)
    x = float(n)
    d = len(n.split('.')[1]) if '.' in n else 0
    if (d < 5 and round(x, d) in num[d]) or (pct and d + 2 < 5 and round(x / 100, d + 2) in num[d + 2]):
        return 'results/logs'
    if n.lstrip('-') in txt:
        return 'docs/config'
    return 'allow' if n in ALLOW else 'UNMATCHED'


def block(k):
    return re.compile(rf'(<!-- BEGIN:{k} -->\n)(?:.*?\n)?(<!-- END:{k} -->)', re.DOTALL)


def put(t, k, v):
    assert len(block(k).findall(t)) == 1, k
    return block(k).sub(lambda m: m.group(1) + v + '\n' + m.group(2), t)


def readme():
    T, derived = tables()
    texts = {}
    for p in DOCS:
        t = p.read_text()
        for k, v in T.items():
            t = put(t, k, v)
        p.write_text(t)
        texts[p.name] = t
    num, txt = sources(derived)
    rep, ok = [], True
    toks = {}
    for name, t in texts.items():
        toks[name] = numbers(t)
        cls = Counter()
        bad, soft = [], Counter()
        for tok, pct in toks[name]:
            c = classify(tok, pct, num, txt)
            cls[c] += 1
            if c == 'UNMATCHED':
                bad.append(tok)
            elif c != 'results/logs':
                soft[f'{tok} ({c})'] += 1
        ok &= not bad
        rep.append(f'- {name}: 수치 토큰 {sum(cls.values())}개 — results/logs 일치 {cls["results/logs"]}, '
                   f'docs·config 일치 {cls["docs/config"]}, 허용 목록 {cls["allow"]}, 불일치 {cls["UNMATCHED"]}'
                   + (f' ({", ".join(bad)})' if bad else ''))
        if name == 'README.md':
            rep.append('  - results/logs 밖 수치: ' + ', '.join(f'{k}×{v}' if v > 1 else k for k, v in sorted(soft.items())))
    a, b = (Counter(norm(t) for t, _ in toks[p.name]) for p in DOCS)
    same = a == b
    ok &= same
    rep.append(f'- 두 파일 수치 토큰 다중집합 일치: {same}' + ('' if same else f' (영어만 {dict(a - b)}, 한국어만 {dict(b - a)})'))
    for k in T:
        blocks = [block(k).search(t).group(0) for t in texts.values()]
        ok &= blocks[0] == blocks[1]
    rep.append(f'- 자동 생성 표 {len(T)}개({", ".join(T)}): 두 파일 동일, CSV에서 생성')
    imgs = [re.findall(r'!\[[^\]]*\]\(([^)]+)\)', t) for t in texts.values()]
    ok &= imgs[0] == imgs[1]
    rep.append(f'- 그림 경로 순서까지 일치: {imgs[0] == imgs[1]} ({len(imgs[0])}개)')
    heads = [re.findall(r'^#{1,3} ', t, re.MULTILINE) for t in texts.values()]
    ok &= heads[0] == heads[1]
    rep.append(f'- 제목 구조(#, ##, ###) 일치: {heads[0] == heads[1]} ({len(heads[0])}개)')
    for name, t in texts.items():
        links = [u.split('#')[0] for u in re.findall(r'\]\(([^)\s]+)\)', t) if not u.startswith(('http', '#', 'mailto'))]
        missing = [u for u in links if not (g.ROOT / u).exists()]
        ok &= not missing
        rep.append(f'- {name} 상대 링크 {len(links)}개, 없는 대상 {len(missing)}' + (f': {missing}' if missing else ''))
    body = '\n'.join(rep)
    print(body)
    cc = g.ROOT / 'docs/code_check.md'
    cc.write_text(put(cc.read_text(), 'readme_check', body))
    assert ok, 'README 대조 실패(위 목록)'


if __name__ == '__main__':
    readme() if sys.argv[1:] == ['readme'] else main()
