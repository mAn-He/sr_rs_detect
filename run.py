"""단계 실행기: 모든 단계 스크립트를 configs/default.yaml envs.gdal 환경의 python으로 subprocess 실행하고
단계별 소요 시간, GPU 사용률, RAM, CPU 사용률을 logs/timing.csv에 기록한다."""
import csv
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load((ROOT / 'configs/default.yaml').read_text())
DU_LIMIT_GB = 150  # 볼륨 200GB, 상한 160GB(CLAUDE.md)에 닿기 전 멈춘다.
CG = Path('/sys/fs/cgroup')
NCPU = 13.6  # cgroup cpu.cfs_quota_us / period (docs/env_report.md)

# stage → [(bash 여부, 명령 인자...)]
STAGES = {
    'data': [('bash', 'scripts/download_data.sh'), ('py', 'src/gdal_ops.py', 'g1')],
    'raster': [('py', 'src/gdal_ops.py', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7')],
    'tiles_labels': [('py', 'src/tiling.py'), ('py', 'src/labels.py')],
    'train_det': [('py', 'src/det_train.py', 'hr'), ('py', 'src/det_train.py', 'lr')],
    # D_LR 수렴 민감도: epochs만 d_lr_sens_epochs로 바꾼 별도 이름 D_LR_e<epochs>.
    'det_sens': [('py', 'src/det_train.py', 'lr_sens')],
    # 학습 후 val 블록에서만 타일 병합 NMS IoU와 운영 conf를 정하고 민감도를 판정한다(results/det_calibration.json).
    'det_calib': [('py', 'src/det_infer.py', 'val'), ('py', 'src/merge.py', 'val'), ('py', 'src/geo_eval.py', 'val')],
    'sr': [('py', 'src/sr_run.py', 'S2_swinir_classical_x4', 'S3_realesrnet_x4plus', 'S4_realesrgan_x4plus',
            'S5_resshift_v3_x4', 'S7_osediff_x4'),
           ('py', 'src/sr_finetune.py'), ('py', 'src/sr_run.py', 'S6_realesrgan_ft_spacenet'),
           ('py', 'src/sr_metrics.py')],
    'eval': [('py', 'src/det_infer.py', 'test'), ('py', 'src/merge.py', 'test'), ('py', 'src/geo_eval.py', 'test'),
             ('py', 'src/report.py')],
}


def du_gb(paths):
    out = subprocess.run(['du', '-sbc', *map(str, paths)], capture_output=True, text=True, check=True)
    return int(out.stdout.splitlines()[-1].split()[0]) / 1e9


# 용량 감시: 프로젝트 밖 폴더(Vla_check, .venvs, miniforge3, weights 등, 단계 실행 중 바뀌지 않음)는 실행 시작 때 한 번만 재고,
# 단계마다 프로젝트 폴더만 du 해서 합으로 판단한다.
FIXED_GB = du_gb([p for p in Path('/workspace').iterdir() if p != ROOT])


def ram_gb():
    """cgroup v1 메모리 사용량에서 비활성 파일 캐시를 뺀 값."""
    st = dict(line.split() for line in (CG / 'memory/memory.stat').read_text().splitlines())
    return (int((CG / 'memory/memory.usage_in_bytes').read_text()) - int(st['total_inactive_file'])) / 1e9


class Monitor(threading.Thread):
    """1초마다 nvidia-smi GPU 사용률·메모리와 cgroup RAM을 기록한다."""

    def __init__(self):
        super().__init__(daemon=True)
        self.gpu, self.gmem, self.ram, self.stop = [], [], [], threading.Event()

    def run(self):
        while not self.stop.wait(1.0):
            q = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, check=True).stdout.split(',')
            self.gpu.append(float(q[0]))
            self.gmem.append(float(q[1]))
            self.ram.append(ram_gb())


cols = ['start', 'stage', 'step', 'seconds', 'smoke', 'workspace_gb', 'project_gb', 'gpu_util_mean', 'gpu_util_max', 'gpu_mem_max_mb',
        'ram_max_gb', 'cpu_util_mean']
for stage in sys.argv[1:]:
    for kind, *cmd in STAGES[stage]:
        project = du_gb([ROOT])
        used = FIXED_GB + project
        assert used < DU_LIMIT_GB, f'/workspace uses {used:.1f} GB >= {DU_LIMIT_GB} GB, stop and report'
        prefix = f"{CFG['miniforge']}/envs/{CFG['envs']['gdal']}"
        exe = 'bash' if kind == 'bash' else f'{prefix}/bin/python'
        # conda 활성화 없이 실행하므로 PROJ 데이터 경로를 명시한다(없으면 osgeo 단독 import 시 EPSG 조회 실패).
        # TORCH_HOME: LPIPS의 AlexNet 가중치를 weights_dir에서 읽는다.
        child_env = os.environ | {'PROJ_DATA': f'{prefix}/share/proj', 'TORCH_HOME': f"{CFG['weights_dir']}/torch"}
        mon = Monitor()
        mon.start()
        cpu0, t0 = int((CG / 'cpuacct/cpuacct.usage').read_text()), time.time()
        start = datetime.now().isoformat(timespec='seconds')
        print(f'[{start}] {stage}: {exe} {" ".join(cmd)}', flush=True)
        subprocess.run([exe, *cmd], cwd=ROOT, env=child_env, check=True)
        sec = time.time() - t0
        cpu = (int((CG / 'cpuacct/cpuacct.usage').read_text()) - cpu0) / 1e9 / sec / NCPU
        mon.stop.set()
        mon.join()
        with open(ROOT / 'logs/timing.csv', 'a', newline='') as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(cols)
            w.writerow([start, stage, ' '.join(cmd), round(sec, 1), CFG['smoke'], round(used, 1), round(project, 2),
                        round(sum(mon.gpu) / max(len(mon.gpu), 1), 1), max(mon.gpu, default=0), max(mon.gmem, default=0),
                        round(max(mon.ram, default=0), 1), round(cpu, 3)])
