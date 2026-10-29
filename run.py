"""단계 실행기: configs/default.yaml의 envs 매핑에 따라 단계별 환경의 python으로 subprocess 실행."""
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load((ROOT / 'configs/default.yaml').read_text())
DU_LIMIT_GB = 150  # 볼륨 200GB, 상한 160GB(CLAUDE.md)에 닿기 전 멈춘다.

# stage → [(envs 키 또는 'bash', 명령 인자...)]
STAGES = {
    'data': [('bash', 'scripts/download_data.sh'), ('gdal', 'src/gdal_ops.py', 'g1')],
    'raster': [('gdal', 'src/gdal_ops.py', 'g2', 'g3', 'g4', 'g5', 'g6')],
}


def du_gb():
    out = subprocess.run(['du', '-sb', '/workspace'], capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0]) / 1e9


for stage in sys.argv[1:]:
    for env, *cmd in STAGES[stage]:
        used = du_gb()
        assert used < DU_LIMIT_GB, f'/workspace uses {used:.1f} GB >= {DU_LIMIT_GB} GB, stop and report'
        prefix = f"{CFG['miniforge']}/envs/{CFG['envs']['gdal' if env == 'bash' else env]}"
        exe = 'bash' if env == 'bash' else f'{prefix}/bin/python'
        # conda 활성화 없이 실행하므로 PROJ 데이터 경로를 명시한다(없으면 osgeo 단독 import 시 EPSG 조회 실패).
        child_env = os.environ | {'PROJ_DATA': f'{prefix}/share/proj'}
        t0 = time.time()
        start = datetime.now().isoformat(timespec='seconds')
        print(f'[{start}] {stage}: {exe} {" ".join(cmd)}', flush=True)
        subprocess.run([exe, *cmd], cwd=ROOT, env=child_env, check=True)
        with open(ROOT / 'logs/timing.csv', 'a', newline='') as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(['start', 'stage', 'step', 'seconds', 'smoke', 'workspace_gb'])
            w.writerow([start, stage, ' '.join(cmd), round(time.time() - t0, 1), CFG['smoke'], round(used, 1)])
