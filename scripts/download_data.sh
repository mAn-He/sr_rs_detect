#!/usr/bin/env bash
# Stage 1: SpaceNet 2 AOI_2_Vegas 칩 위치 목록 작성(헤더만 읽음). 여러 번 실행해도 안전하다.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cfg() { sed -n "s/^$1: *//p" "$ROOT/configs/default.yaml"; }
G=$(cfg miniforge)/envs/geo/bin
NP=$(cfg n_proc)
S3=s3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_2_Vegas
D=$ROOT/data
mkdir -p "$D"
export AWS_NO_SIGN_REQUEST=YES GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

# 칩 위치 목록: PS-RGB 헤더의 geotransform만 /vsis3로 읽는다(파일은 받지 않는다).
if [ ! -s "$D/chip_index.csv" ]; then
  "$G/aws" s3 ls --no-sign-request "$S3/PS-RGB/" | awk '{print $4}' | sed -n 's/.*_PS-RGB_img\([0-9]*\)\.tif$/\1/p' > "$D/chip_ids.txt"
  "$G/python" - "$D" "${S3#s3://}" "$NP" <<'PYEOF'
import sys
from concurrent.futures import ThreadPoolExecutor

from osgeo import gdal

d, s3, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
gdal.UseExceptions()
ids = [int(x) for x in open(f'{d}/chip_ids.txt').read().split()]


def header(cid):
    ds = gdal.Open(f'/vsis3/{s3}/PS-RGB/SN2_buildings_train_AOI_2_Vegas_PS-RGB_img{cid}.tif')
    x0, dx, _, y0, _, dy = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    return cid, x0, y0, x0 + w * dx, y0 + h * dy, w, h


with ThreadPoolExecutor(n) as ex:
    rows = sorted(ex.map(header, ids))
with open(f'{d}/chip_index.csv', 'w') as f:
    f.write('chip_id,minx,maxy,maxx,miny,width,height\n')
    for r in rows:
        f.write(','.join(map(str, r)) + '\n')
print(len(rows), 'chips indexed')
PYEOF
fi

# 창 선택: 칩 격자에서 train 칩 n_chips±50장을 담는 가장 조밀한 직사각형(동률이면 칩 수가 많은 쪽).
if [ ! -s "$D/window_chips.txt" ]; then
  "$G/python" - "$D" "$(cfg n_chips)" <<'PYEOF'
import json
import sys

import numpy as np
import pandas as pd

d, target = sys.argv[1], int(sys.argv[2])
df = pd.read_csv(f'{d}/chip_index.csv')
step = df.maxx.iloc[0] - df.minx.iloc[0]
col = np.rint((df.minx - df.minx.min()) / step).astype(int)
row = np.rint((df.maxy.max() - df.maxy) / step).astype(int)
assert np.allclose(df.minx.min() + col * step, df.minx) and np.allclose(df.maxy.max() - row * step, df.maxy), 'chips are not on one grid'
occ = np.zeros((row.max() + 1, col.max() + 1), int)
occ[row, col] = 1
cs = np.pad(occ.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
best = None
for h in range(1, occ.shape[0] + 1):
    for w in range(1, occ.shape[1] + 1):
        n = cs[h:, w:] - cs[:-h, w:] - cs[h:, :-w] + cs[:-h, :-w]
        for i, j in np.argwhere(abs(n - target) <= 50):
            key = (n[i, j] / (h * w), n[i, j], -i, -j)
            if best is None or key > best[0]:
                best = (key, int(i), int(j), h, w)
(dens, n, _, _), i, j, h, w = best
sel = df[(row >= i) & (row < i + h) & (col >= j) & (col < j + w)]
assert len(sel) == n
sel.chip_id.to_csv(f'{d}/window_chips.txt', index=False, header=False)
info = dict(grid_rows=int(occ.shape[0]), grid_cols=int(occ.shape[1]), row0=i, col0=j, rows=h, cols=w, n_chips=int(n),
            density=round(float(dens), 4), lon_min=float(sel.minx.min()), lon_max=float(sel.maxx.max()),
            lat_min=float(sel.miny.min()), lat_max=float(sel.maxy.max()))
json.dump(info, open(f'{d}/window.json', 'w'), indent=1)
print(info)
PYEOF
fi

# 다운로드: 창 안 train 칩의 PS-RGB, MS, 건물 GeoJSON만. aws 한 프로세스에 --include 목록을 넘긴다
# (FUSE 볼륨에서 aws CLI 시작에 파일당 약 9 s가 걸려 파일별 호출은 쓰지 않는다). 이미 받은 파일은 건너뛴다.
for p in PS-RGB:tif MS:tif geojson_buildings:geojson; do
  f=${p%%:*}; ext=${p##*:}
  mkdir -p "$D/raw/$f"
  mapfile -t inc < <(while read -r id; do
    [ -s "$D/raw/$f/SN2_buildings_train_AOI_2_Vegas_${f}_img$id.$ext" ] || printf -- '--include\n*_img%s.%s\n' "$id" "$ext"
  done < "$D/window_chips.txt")
  [ ${#inc[@]} -eq 0 ] || "$G/aws" s3 cp --no-sign-request --only-show-errors --recursive "$S3/$f/" "$D/raw/$f/" --exclude '*' "${inc[@]}"
  n=$(ls "$D/raw/$f" | wc -l)
  [ "$n" -eq "$(wc -l < "$D/window_chips.txt")" ] || { echo "$f: $n files, expected $(wc -l < "$D/window_chips.txt")"; exit 1; }
  echo "$f: $n files, $(du -sh "$D/raw/$f" | cut -f1)"
done
