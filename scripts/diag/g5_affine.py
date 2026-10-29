"""원본 MS 격자에서 ECC 아핀 추정(스케일·이동)과 칩 내부 좌우·상하 절반 오프셋.

변형: orig(메타데이터 그대로), h4(MS 화소 동서·남북 모두 4×PS), xfix(동서만 4×PS, 파이프라인 정정과 같음).
사용: python scripts/diag/g5_affine.py <out_dir>  (<out_dir>/chips.csv의 칩 목록) → <out_dir>/affine.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g5_investigate import native, norm, pc

CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7)


def job(cid):
    r = {'chip_id': cid}
    for tag, kw in (('orig', {}), ('h4', {'px_x': 4 * 2.7e-6, 'px_y': 4 * 2.7e-6}), ('xfix', {'px_x': 4 * 2.7e-6})):
        a, b = native(cid, **kw)
        a, b = a[4:-4, 4:-4], b[4:-4, 4:-4]
        w = np.eye(2, 3, dtype=np.float32)
        try:
            _, w = cv2.findTransformECC(norm(a), norm(b), w, cv2.MOTION_AFFINE, CRIT, None, 5)
            n = a.shape[0]
            c = (n - 1) / 2
            # 칩 중심에서의 이동, 그리고 칩 폭(155 px)에 걸친 이동 변화(스케일)
            r[f'{tag}_tx_center'] = float(w[0, 0] * c + w[0, 1] * c + w[0, 2] - c)
            r[f'{tag}_ty_center'] = float(w[1, 0] * c + w[1, 1] * c + w[1, 2] - c)
            r[f'{tag}_sx_px_over_chip'] = float((w[0, 0] - 1) * (n - 1))
            r[f'{tag}_sy_px_over_chip'] = float((w[1, 1] - 1) * (n - 1))
        except cv2.error:
            r[f'{tag}_fail'] = 1
        h = a.shape[1] // 2
        r[f'{tag}_left_dx'], _ = pc(a[:, :h], b[:, :h])
        r[f'{tag}_right_dx'], _ = pc(a[:, h:], b[:, h:])
        _, r[f'{tag}_top_dy'] = pc(a[:h], b[:h])
        _, r[f'{tag}_bottom_dy'] = pc(a[h:], b[h:])
    return r


if __name__ == '__main__':
    ids = pd.read_csv(f'{sys.argv[1]}/chips.csv').chip_id.tolist()
    with ProcessPoolExecutor(12) as ex:
        pd.DataFrame(list(ex.map(job, ids, chunksize=8))).to_csv(f'{sys.argv[1]}/affine.csv', index=False)
    print('done')
