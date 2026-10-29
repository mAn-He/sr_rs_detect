"""G5 블록 오프셋 지도: MS 동서 화소 정정 전/후 g5_registration CSV를 나란히 그린다(phaseCorrelate 화살표, 색 = 크기).

사용: python scripts/diag/g5_map.py <before.csv> <after.csv> <out.png>  (현재 data/<mode>/splits.geojson 블록 기준)
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import gdal_ops as g

K = 800  # 화살표 배율: 1 LR px를 800 m로 그린다.
blocks = gpd.read_file(g.W / 'splits.geojson')
fp = gpd.read_file(g.W / 'footprint.geojson')
norm = matplotlib.colors.Normalize(0, 0.8)
fig, ax = plt.subplots(1, 2, figsize=(17, 8.5), constrained_layout=True)
for a, path, title in zip(ax, sys.argv[1:3], ('before MS x-pixel fix', 'after MS x-pixel fix')):
    d = blocks.merge(pd.read_csv(path), on=['block_id', 'split'])
    fp.boundary.plot(ax=a, color='0.75', lw=0.3)
    d.plot(ax=a, column='shift', cmap='viridis', norm=norm, alpha=0.85, edgecolor='0.4', lw=0.2)
    c = d.geometry.centroid
    a.quiver(c.x, c.y, d.dx * K, -d.dy * K, angles='xy', scale_units='xy', scale=1, width=0.003, color='k')
    over = d[d['shift'] > g.CFG['g5_max_shift']]
    over.boundary.plot(ax=a, color='red', lw=1.5)
    a.set_title(f'{title}: phaseCorrelate (even DFT)\n'
                f'dx median {d.dx.median():+.3f}, dy median {d.dy.median():+.3f}, shift max {d["shift"].max():.3f} LR px; '
                f'>{g.CFG["g5_max_shift"]}: {len(over)}/{len(d)} (red)', fontsize=10)
    a.set_aspect('equal')
    a.set_xticks([])
    a.set_yticks([])
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, shrink=0.6, label='offset magnitude (LR px, 1.2 m)')
fig.suptitle('G5 HR(4x4 mean) vs LR offset per 576 m block. Arrow = LR content displacement, 1 LR px drawn as 800 m')
fig.savefig(sys.argv[3], dpi=110)
print('saved', sys.argv[3])
