"""격자 정렬, 픽셀↔지리 좌표 변환, OBB 라벨 변환 테스트(작은 합성 배열)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_origin
from shapely.geometry import Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import gdal_ops as g
import labels
import tiling


class Src:
    """window()가 쓰는 transform만 가진 합성 래스터."""

    def __init__(self, res):
        self.transform = from_origin(600000.0, 4000000.0, res, res)


def test_window_on_grid():
    w = g.window(Src(g.LR), box(600000.0 + 1.2 * 10, 4000000.0 - 1.2 * 30, 600000.0 + 1.2 * 490, 4000000.0 - 1.2 * 20))
    assert (w.col_off, w.row_off, w.width, w.height) == (10, 20, 480, 10)
    hw = g.window(Src(g.HR), box(600000.0 + 1.2 * 10, 4000000.0 - 1.2 * 30, 600000.0 + 1.2 * 490, 4000000.0 - 1.2 * 20))
    assert (hw.col_off, hw.row_off, hw.width, hw.height) == (40, 80, 1920, 40)


def test_window_off_grid_fails():
    with pytest.raises(AssertionError):
        g.window(Src(g.LR), box(600000.5, 3999000.0, 600012.5, 4000000.0))


def test_tiles_cover_block_on_grid():
    import geopandas as gpd
    blk = box(600000.0, 4000000.0 - g.B, 600000.0 + g.B, 4000000.0)
    blocks = gpd.GeoDataFrame({'block_id': [0], 'split': ['train']}, geometry=[blk], crs=g.UTM)
    t = tiling.tile_windows(blocks, blk)
    assert len(t) == 4 and t.valid_frac.eq(1).all()
    assert np.isclose(t.union_all().area, blk.area)
    for geom in t.geometry:
        x0, _y0, x1, y1 = geom.bounds
        assert np.isclose(x1 - x0, 307.2) and np.allclose(np.array([x0, y1]) / g.LR, np.rint(np.array([x0, y1]) / g.LR))


def test_pixel_geo_roundtrip_and_sr_transform():
    lr = from_origin(600000.0, 4000000.0, g.LR, g.LR)
    hr = lr * Affine.scale(1 / 4)
    assert hr.almost_equals(from_origin(600000.0, 4000000.0, g.HR, g.HR), 1e-9)
    x, y = lr @ (10, 20)
    assert (x, y) == (600012.0, 3999976.0)
    assert np.allclose(~hr @ (x, y), (40, 80))


def test_obb_label_rotated_rectangle():
    tr = from_origin(600000.0, 4000000.0, g.HR, g.HR)
    # 타일 픽셀 (100,200)-(300,260) 사각형을 30° 회전한 폴리곤
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    px = np.array([[-100, -30], [100, -30], [100, 30], [-100, 30]]) @ np.array([[c, s], [-s, c]]) + [200, 230]
    poly = Polygon([tr @ tuple(p) for p in px])
    pts = labels.obb_label(poly, tr, 1024) * 1024
    assert np.allclose(np.array(sorted(map(tuple, pts))), np.array(sorted(map(tuple, px))), atol=1e-2)


def test_obb_label_clipped_to_unit():
    tr = from_origin(600000.0, 4000000.0, g.HR, g.HR)
    poly = Polygon([tr @ p for p in [(-10, -10), (50, -10), (50, 50), (-10, 50)]])
    pts = labels.obb_label(poly, tr, 1024)
    assert pts.min() == 0 and pts.max() <= 1


def test_write_tif_roundtrip(tmp_path):
    t = from_origin(600000.0, 4000000.0, g.HR, g.HR)
    img = np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8)
    g.write_tif(tmp_path / 'a.tif', img, t)
    with rasterio.open(tmp_path / 'a.tif') as src:
        assert src.transform == t and (src.read() == img).all() and src.profile['compress'] == 'deflate'
