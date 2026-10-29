"""G5 가설 검증: MS 축척 정정의 기준점(A), 남북 축척(B), 칩 위상(C).

원본 MS 격자에서 칩마다 ECC 아핀(중심 이동·칩 폭 스케일 변화)과 좌우·상하 절반 phaseCorrelate를 잰다.
변형(MS geotransform 가설, PS 화소 2.7e-6° 단위):
  orig   메타데이터 그대로
  A      동서 화소 4×PS, 칩 중심 고정(원점 서쪽 1 PS px)
  AB     A + 남북 화소 4×PS, 칩 중심 고정(원점 북쪽 1 PS px)
  C      동서·남북 화소 4×PS, 원점 = 칩 원점 − 2 PS px × 위상. 위상은 전역 칩 격자 열(행) 번호 홀짝으로 정한다
         (650 PS px ÷ 4 = 162.5라 원래 MS 격자에 대한 칩 원점 위상이 칩마다 0 / 0.5 MS px로 번갈아 바뀐다는 가설).
         어느 홀짝이 위상 0.5인지는 C_odd(홀수 열·행이 0.5), C_even(짝수가 0.5) 두 경우를 모두 잰다.
사용: python scripts/diag/g5_phase.py <out_dir> → <out_dir>/phase.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g5_investigate import REPO, native, norm, pc

PS = 2.7e-6
CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7)


def measure(cid, **kw):
    a, b = native(cid, **kw)
    a, b = a[4:-4, 4:-4], b[4:-4, 4:-4]
    n = a.shape[0]
    c = (n - 1) / 2
    _, w = cv2.findTransformECC(norm(a), norm(b), np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE, CRIT, None, 5)
    h = n // 2
    return {'tx': float(w[0, 0] * c + w[0, 1] * c + w[0, 2] - c), 'ty': float(w[1, 0] * c + w[1, 1] * c + w[1, 2] - c),
            'sx': float((w[0, 0] - 1) * (n - 1)), 'sy': float((w[1, 1] - 1) * (n - 1)),
            'left_dx': pc(a[:, :h], b[:, :h])[0], 'right_dx': pc(a[:, h:], b[:, h:])[0],
            'top_dy': pc(a[:h], b[:h])[1], 'bottom_dy': pc(a[h:], b[h:])[1]}


def job(args):
    cid, col, row = args
    r = {'chip_id': cid, 'col': col, 'row': row}
    v = {'orig': {}, 'A': {'px_x': 4 * PS, 'shift_x_ps': 1},
         'AB': {'px_x': 4 * PS, 'px_y': 4 * PS, 'shift_x_ps': 1, 'shift_y_ps': 1}}
    for tag, odd in (('C_odd', 1), ('C_even', 0)):
        v[tag] = {'px_x': 4 * PS, 'px_y': 4 * PS, 'shift_x_ps': 2 * (col % 2 == odd), 'shift_y_ps': 2 * (row % 2 == odd)}
    for tag, kw in v.items():
        r.update({f'{tag}_{k}': x for k, x in measure(cid, **kw).items()})
    return r


if __name__ == '__main__':
    idx = pd.read_csv(REPO / 'data/chip_index.csv')
    step = idx.maxx.iloc[0] - idx.minx.iloc[0]
    idx['col'] = np.rint((idx.minx - idx.minx.min()) / step).astype(int)
    idx['row'] = np.rint((idx.maxy.max() - idx.maxy) / step).astype(int)
    ids = [int(x) for x in (REPO / 'data/window_chips.txt').read_text().split()]
    sel = idx.set_index('chip_id').loc[ids]
    with ProcessPoolExecutor(12) as ex:
        df = pd.DataFrame(list(ex.map(job, zip(sel.index, sel.col, sel.row), chunksize=8)))
    df.to_csv(Path(sys.argv[1]) / 'phase.csv', index=False)
    print('done', len(df))
