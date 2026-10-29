"""geo 환경 검증: import, GPU, GDAL 드라이버, 사용할 OpenCV API, YOLO26-OBB 공식 최소 추론, LPIPS 로드."""
from importlib.metadata import version
from pathlib import Path

import cv2
import geopandas
import lpips
import matplotlib
import numpy as np
import pandas
import pyproj
import rasterio
import scipy
import shapely
import skimage
import torch
import torchvision
import ultralytics
import yaml
from osgeo import gdal
from ultralytics import YOLO
from ultralytics.utils import ASSETS

cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/default.yaml').read_text())
print(f'{"gdal":12s} {gdal.__version__}')
for m in (rasterio, geopandas, shapely, pyproj, cv2, torch, torchvision, ultralytics, skimage, scipy, pandas, matplotlib,
          np):
    print(f'{m.__name__:12s} {m.__version__}')
print(f'{"lpips":12s} {version("lpips")}')
print('torch cuda', torch.version.cuda, torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
for d in ('VRT', 'GTiff', 'COG', 'MEM'):
    assert gdal.GetDriverByName(d) is not None, d

# 파이프라인에서 쓰는 OpenCV API를 작은 합성 배열로 호출한다.
a = np.zeros((64, 64), np.float32)
a[20:30, 20:30] = 1
b = np.roll(a, (2, 3), axis=(0, 1))
(dx, dy), _ = cv2.phaseCorrelate(a, b)
assert abs(dx - 3) < 0.1 and abs(dy - 2) < 0.1, (dx, dy)
rect = cv2.minAreaRect(np.array([[0, 0], [10, 0], [10, 5], [0, 5]], np.float32))
assert cv2.boxPoints(rect).shape == (4, 2)
keep = cv2.dnn.NMSBoxesRotated([rect, rect], [0.9, 0.8], 0.1, 0.5)
assert len(keep) == 1
u8 = (a * 255).astype(np.uint8)
assert cv2.Canny(u8, 50, 150).any()
assert cv2.dilate(u8, np.ones((3, 3), np.uint8)).sum() > u8.sum()
assert cv2.Laplacian(u8, cv2.CV_64F).var() > 0

model = YOLO(f'{cfg["weights_dir"]}/yolo/yolo26s-obb.pt')
r = model.predict(ASSETS / 'bus.jpg', device=0, verbose=False)[0]
print('yolo26s-obb bus.jpg: obb', len(r.obb), 'speed', r.speed)
assert r.obb is not None

lpips.LPIPS(net='alex').cuda()
print('VERIFY geo OK')
