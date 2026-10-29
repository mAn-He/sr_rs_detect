"""G5 모의: MS 동서 원점·화소 결정적 규칙을 data/<mode>를 바꾸지 않고 별도 폴더 VRT로 적용해 G5를 잰다.

규칙(동서만): 원래 MS 격자 원점 = 전역 칩 격자 서쪽 끝, 화소 = 4×PS. 칩 열 k의 원점은 650k PS px이므로
MS 칩 원점 = 칩 원점 − 2 PS px × (k mod 2), 동서 화소 = 4×PS. 남북은 메타데이터 그대로(cx).
변형 cxb: cx + 남북 화소 4×PS, 칩 중심 고정(원점 북쪽 1 PS px, 가설 B).
사용: python scripts/diag/g5_rule_sim.py <out_dir> <cx|cxb>  → <out_dir>/sim_<variant>/g5_registration.csv
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import gdal_ops as g

variant = sys.argv[2]
assert variant in ('cx', 'cxb')
out = Path(sys.argv[1]) / f'sim_{variant}'
fix = out / 'ms_fix'
fix.mkdir(parents=True, exist_ok=True)
idx = pd.read_csv(g.D / 'chip_index.csv')
step = idx.maxx.iloc[0] - idx.minx.iloc[0]
col = dict(zip(idx.chip_id, np.rint((idx.minx - idx.minx.min()) / step).astype(int)))
for c in g.window_ids():
    ps, ms = gdal.Open(str(g.chip_path('PS-RGB', c))), gdal.Open(str(g.chip_path('MS', c)))
    gp, gm = ps.GetGeoTransform(), ms.GetGeoTransform()
    x0 = gm[0] - 2 * gp[1] * (col[c] % 2)
    y0, py = (gm[3] - gp[5], 4 * gp[5]) if variant == 'cxb' else (gm[3], gm[5])
    ullr = (x0, y0, x0 + 4 * gp[1] * ms.RasterXSize, y0 + py * ms.RasterYSize)
    gdal.Translate(str(fix / f'{c}.vrt'), ms, format='VRT', outputBounds=ullr)
(out / 'ms_src.vrt.txt').write_text('\n'.join(str(fix / f'{c}.vrt') for c in g.window_ids()) + '\n')
g.run(['gdalbuildvrt', '-overwrite', '-input_file_list', out / 'ms_src.vrt.txt', out / 'ms_src.vrt'])
te = __import__('json').loads((g.W / 'extent.json').read_text())['te']
common = ['gdalwarp', '-overwrite', '-of', 'VRT', '-t_srs', g.UTM, '-te', *te, '-r', 'cubic']
g.run(common + ['-tr', g.HR, g.HR, g.W / 'hr_src.vrt', out / 'hr_utm.vrt'])
g.run(common + ['-tr', g.LR, g.LR, *[x for b in g.CFG['ms_rgb_bands'] for x in ('-b', b)], out / 'ms_src.vrt', out / 'lr_utm.vrt'])
for f in ('splits.geojson', 'chips.geojson'):
    shutil.copy(g.W / f, out / f)
g.W = out
try:
    g.g5()
    print('SIM G5: all blocks pass')
except AssertionError as e:
    print('SIM G5 FAIL:', str(e).splitlines()[0])
