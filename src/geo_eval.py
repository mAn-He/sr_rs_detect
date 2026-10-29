"""G12 지리 평가(test 블록, UTM): 회전 IoU ≥ match_iou 매칭, AP50, 면적 구간별 재현율, 거짓 양성,
test 블록 부트스트랩 95% CI, gain·recovery·verdict(경로 1 대비). 평가 영역은 footprint를 안쪽으로 제외 폭만큼 줄인
영역 ∩ test 블록이며, 면적의 eval_outside_frac 이상이 밖인 GT·예측은 모두 제외한다(M4). 제외 폭 0/8/16 LR px(M5)."""
import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.geometry import Polygon

import gdal_ops as g
from det_infer import rows

EDGES = [0, *g.CFG['area_bins'], np.inf]
BINS = ['lt50' if i == 0 else f'gt{int(EDGES[i])}' if np.isinf(EDGES[i + 1]) else f'{int(EDGES[i])}_{int(EDGES[i + 1])}'
        for i in range(len(EDGES) - 1)]


def obb(geom):
    """GT 폴리곤의 UTM 최소 회전 사각형(float32 정밀도를 위해 국소 원점에서 계산)."""
    parts = getattr(geom, 'geoms', [geom])
    xy = np.vstack([np.asarray(p.exterior.coords)[:, :2] for p in parts])
    o = xy.min(axis=0)
    return Polygon(cv2.boxPoints(cv2.minAreaRect((xy - o).astype(np.float32))).astype(np.float64) + o)


def inside(geoms, region):
    return np.array([gm.intersection(region).area / gm.area for gm in geoms]) > 1 - g.CFG['eval_outside_frac']


def match(pred, gt):
    """conf 내림차순 탐욕 매칭. 반환: 예측별 TP 여부, GT별 매칭된 예측 conf(없으면 -1)."""
    order = np.argsort(-pred.conf.to_numpy(), kind='stable')
    tree = STRtree(gt)
    used = np.zeros(len(gt), bool)
    gt_conf = np.full(len(gt), -1.0)
    tp = np.zeros(len(pred), bool)
    pg, pc = pred.geometry.to_numpy(), pred.conf.to_numpy()
    for i in order:
        best, bj = g.CFG['match_iou'], -1
        for j in tree.query(pg[i], predicate='intersects'):
            if used[j]:
                continue
            inter = pg[i].intersection(gt[j]).area
            iou = inter / (pg[i].area + gt[j].area - inter)
            if iou >= best:
                best, bj = iou, j
        if bj >= 0:
            used[bj], tp[i], gt_conf[bj] = True, True, pc[i]
    return tp, gt_conf


def ap50(conf, tp, w, n_gt):
    """Ultralytics compute_ap과 같은 101점 보간 AP. w는 부트스트랩 블록 가중치."""
    if n_gt <= 0:
        return np.nan
    o = np.argsort(-conf, kind='stable')
    ctp, cfp = np.cumsum(w[o] * tp[o]), np.cumsum(w[o] * ~tp[o])
    rec = np.concatenate([[0.0], ctp / n_gt, [1.0]])
    pre = np.concatenate([[1.0], ctp / np.maximum(ctp + cfp, 1e-12), [0.0]])
    pre = np.flip(np.maximum.accumulate(np.flip(pre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, rec, pre), x))


def stats(d, wb):
    """블록 가중치 wb(블록 수 길이)로 AP50과 면적 구간별 재현율(운영점 conf)."""
    wp, wg = wb[d['p_blk']], wb[d['g_blk']]
    out = {'AP50': ap50(d['conf'], d['tp'], wp, wg.sum())}
    rec = d['g_conf'] >= g.CFG['det_conf_op']
    for k, b in enumerate(BINS):
        m = d['g_bin'] == k
        out[f'recall_{b}'] = (wg[m] * rec[m]).sum() / wg[m].sum() if wg[m].sum() > 0 else np.nan
    return out


def block_of(geoms, test):
    pts = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geoms, crs=g.UTM).centroid)
    j = gpd.sjoin_nearest(pts, test[['geometry']].reset_index(drop=True))
    return j.groupby(level=0).index_right.first().reindex(pts.index).to_numpy()


def main():
    blocks = g.used_blocks()
    test = blocks[blocks.split == 'test'].reset_index(drop=True)
    union = test.union_all()
    fp = gpd.read_file(g.W / 'footprint.geojson').geometry.iloc[0]
    bld = gpd.read_file(g.W / 'buildings_utm.geojson')
    bld = bld[bld.intersects(union)].reset_index(drop=True)
    names = [r[0] for r in rows()]
    preds = {n: gpd.read_file(g.W / 'preds' / f'{n}.geojson').to_crs(g.UTM) for n in names}
    n_boot = g.CFG['smoke_bootstrap'] if g.CFG['smoke'] else g.CFG['bootstrap']
    rng = np.random.default_rng(g.CFG['seed'])
    boot = rng.integers(0, len(test), size=(n_boot, len(test)))
    W_boot = np.stack([np.bincount(b, minlength=len(test)) for b in boot]).astype(float)
    out = []
    for buf in g.CFG['eval_buffer_sens']:
        region = fp.buffer(-buf * g.LR).intersection(union)
        gt = bld[inside(bld.geometry, region)].reset_index(drop=True)
        gt_obb = np.array([obb(x) for x in gt.geometry])
        g_blk = block_of(gt.geometry, test)
        g_bin = np.digitize(gt.area_m2, EDGES[1:-1])
        res, bs = {}, {}
        for n in names:
            p = preds[n]
            p = p[inside(p.geometry, region)].reset_index(drop=True)
            tp, g_conf = match(p, gt_obb)
            d = {'conf': p.conf.to_numpy(), 'tp': tp, 'p_blk': block_of(p.geometry, test), 'g_blk': g_blk,
                 'g_conf': g_conf, 'g_bin': g_bin}
            res[n] = stats(d, np.ones(len(test)))
            res[n].update({'false_positives': int(((d['conf'] >= g.CFG['det_conf_op']) & ~tp).sum()),
                           'n_gt': len(gt), 'n_pred_op': int((d['conf'] >= g.CFG['det_conf_op']).sum())})
            bs[n] = pd.DataFrame([stats(d, w) for w in W_boot])
        hr, lr = res['HR']['AP50'], res['LR']['AP50']
        for n in names:
            r = {'method': n, 'buffer_lr': buf, **res[n]}
            for k in bs[n].columns:
                r[f'{k}_ci_low'], r[f'{k}_ci_high'] = np.nanpercentile(bs[n][k], [2.5, 97.5])
            if n not in ('HR', 'LR'):
                for k in bs[n].columns:
                    gk = 'gain' if k == 'AP50' else f'gain_{k}'
                    r[gk] = res[n][k] - res['LR'][k]
                    diff = bs[n][k] - bs['LR'][k]
                    lo, hi = np.nanpercentile(diff, [2.5, 97.5])
                    r[f'{gk}_ci_low'], r[f'{gk}_ci_high'] = lo, hi
                    r['verdict' if k == 'AP50' else f'verdict_{k}'] = \
                        'useful' if lo > 0 else 'harmful' if hi < 0 else 'inconclusive'
                r['recovery'] = r['gain'] / (hr - lr) if hr != lr else np.nan
            out.append(r)
        print(f'buffer {buf} LR px: {len(gt)} GT,', {n: round(res[n]['AP50'], 4) for n in names})
    df = pd.DataFrame(out)
    df.to_csv(g.ROOT / 'results/geo_eval.csv', index=False)
    print(df[['method', 'buffer_lr', 'AP50', 'AP50_ci_low', 'AP50_ci_high', 'false_positives']].to_string())


if __name__ == '__main__':
    main()
