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
