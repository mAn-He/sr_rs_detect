"""SR 화질·OpenCV 엣지 지표(test 타일, 평가 영역 마스크 안, M6): PSNR, SSIM(맵의 마스크 평균), LPIPS(spatial=True 맵의
마스크 평균), edge_precision/recall(cv2.Canny, 1픽셀 cv2.dilate), sharpness_ratio(cv2.Laplacian 분산 SR/HR).
Canny 임계값은 HR train 타일 밝기 채널(유효 픽셀) 중앙값 × canny_ratio로 한 번 정해 모든 방법에 쓴다."""
import json

import cv2
import geopandas as gpd
import lpips
import numpy as np
import pandas as pd
import rasterio
import torch
from skimage.metrics import structural_similarity

import gdal_ops as g
from sr_run import ENABLED


def gray(img):
    return cv2.cvtColor(np.ascontiguousarray(img.transpose(1, 2, 0)), cv2.COLOR_RGB2GRAY)


def canny_thresholds(tiles, fp):
    vals = []
    for t in tiles[(tiles.split == 'train') & tiles.use].itertuples():
        with rasterio.open(g.W / f'tiles/hr/images/train/{t.tile_id}.tif') as src:
            img, tr = src.read(), src.transform
        vals.append(gray(img)[g.tile_mask(tr, img.shape[1:], fp)])
    med = float(np.median(np.concatenate(vals)))
    lo, hi = (round(r * med) for r in g.CFG['canny_ratio'])
    return {'median_luma': med, 'low': lo, 'high': hi}


def main():
    tiles = gpd.read_file(g.W / 'tiles.geojson')
    fp = gpd.read_file(g.W / 'footprint.geojson').geometry.iloc[0]
    canny = canny_thresholds(tiles, fp)
    (g.ROOT / 'results/canny_thresholds.json').write_text(json.dumps(canny, indent=1))
    print('Canny', canny)
    region = fp.buffer(-g.CFG['eval_buffer_lr'] * g.LR)
    test = tiles[(tiles.split == 'test') & tiles.use]
    net = lpips.LPIPS(net='alex', spatial=True).cuda().eval()
    k = np.ones((3, 3), np.uint8)
    out = []
    for m in g.CFG['sr_methods']:
        if not ENABLED.get(m, True):
            continue
        acc = dict.fromkeys(['sse', 'ssim', 'lpips', 'n', 'n3', 'tp_p', 'sr_e', 'tp_r', 'hr_e'], 0.0)
        lap = {'sr': [], 'hr': []}
        for t in test.itertuples():
            with rasterio.open(g.W / f'tiles/hr/images/test/{t.tile_id}.tif') as h, \
                    rasterio.open(g.W / 'sr' / m / f'images/test/{t.tile_id}.tif') as s:
                assert h.transform == s.transform and h.shape == s.shape
                hr, sr, tr = h.read(), s.read(), h.transform
            mask = g.tile_mask(tr, hr.shape[1:], region)
            if not mask.any():
                continue
            d = hr.astype(np.float64) - sr
            acc['sse'] += (d[:, mask] ** 2).sum()
            acc['n'] += mask.sum()
            acc['n3'] += 3 * mask.sum()
            _, smap = structural_similarity(hr.transpose(1, 2, 0), sr.transpose(1, 2, 0), channel_axis=2, data_range=255,
                                            full=True)
            acc['ssim'] += smap.mean(axis=2)[mask].sum()
            with torch.no_grad():
                a, b = (torch.from_numpy(x).float().div(127.5).sub(1).unsqueeze(0).cuda() for x in (hr, sr))
                lmap = net(a, b)[0, 0].cpu().numpy()
            acc['lpips'] += lmap[mask].sum()
            gh, gs = gray(hr), gray(sr)
            eh, es = cv2.Canny(gh, canny['low'], canny['high']) > 0, cv2.Canny(gs, canny['low'], canny['high']) > 0
            dh, ds = cv2.dilate(eh.astype(np.uint8), k) > 0, cv2.dilate(es.astype(np.uint8), k) > 0
            acc['tp_p'] += (es & dh & mask).sum()
            acc['sr_e'] += (es & mask).sum()
            acc['tp_r'] += (eh & ds & mask).sum()
            acc['hr_e'] += (eh & mask).sum()
            lap['hr'].append(cv2.Laplacian(gh, cv2.CV_64F)[mask])
            lap['sr'].append(cv2.Laplacian(gs, cv2.CV_64F)[mask])
        out.append({'method': m, 'PSNR': 10 * np.log10(255**2 / (acc['sse'] / acc['n3'])), 'SSIM': acc['ssim'] / acc['n'],
                    'LPIPS': acc['lpips'] / acc['n'], 'edge_precision': acc['tp_p'] / acc['sr_e'],
                    'edge_recall': acc['tp_r'] / acc['hr_e'],
                    'sharpness_ratio': np.concatenate(lap['sr']).var() / np.concatenate(lap['hr']).var(),
                    'n_eval_px': int(acc['n'])})
        print(out[-1])
    pd.DataFrame(out).to_csv(g.ROOT / 'results/sr_metrics.csv', index=False)


if __name__ == '__main__':
    main()
