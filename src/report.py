"""results/summary.csv(행: HR 상한, 경로 1, 경로 2 SR 방법), 화질 지표 순위와 AP50 순위의 Spearman 상관, 그림 (a)~(f)."""
import cv2
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import spearmanr

import gdal_ops as g
from det_infer import rows
from geo_eval import BINS, inside, match, obb

matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = g.ROOT / 'results'
INK, MUTED, SERIES = '#0b0b0b', '#52514e', ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
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
        ax.annotate(r.method.split('_')[0], (getattr(r, x), r.AP50_geo), xytext=(6, 4), textcoords='offset points',
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


def main():
    df = summary()
    spearman(df)
    scatter(df, 'PSNR', 'report_a_psnr_vs_ap50.png', 'PSNR (dB, evaluation mask)')
    recall_bars(df)
    scenes(df)
    scatter(df, 'sr_sec_per_km2', 'report_e_time_vs_ap50.png', 'SR time (s per km², excl. model load)', logx=True)
    scatter(df, 'sharpness_ratio', 'report_f_sharpness_vs_ap50.png', 'sharpness_ratio (Laplacian var SR/HR)')


if __name__ == '__main__':
    main()
