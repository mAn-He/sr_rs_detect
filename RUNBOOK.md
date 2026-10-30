# RUNBOOK

실제로 실행해 성공한 명령만 순서대로 기록한다. 모든 명령은 `/workspace/rs-sr-detect`에서 실행한다.

## Stage 0 env

```bash
# 2026-10-29: 환경 생성은 이 명령으로 성공. 가중치 단계는 HF_HUB_ENABLE_HF_TRANSFER=0 수정 후 그 단계만 재실행해 성공.
# hf_transfer 수정 후 전체 재실행 미검증.
T=$(date +%Y%m%d_%H%M); nohup bash scripts/setup_envs.sh > logs/stage0_setup_$T.log 2>&1 &
/workspace/miniforge3/envs/geo/bin/python scripts/verify_env_geo.py
/workspace/miniforge3/envs/sr/bin/python scripts/verify_env_sr.py
/workspace/miniforge3/envs/osediff/bin/python scripts/verify_env_osediff.py
CONDA_PKGS_DIRS=/workspace/cache/conda /workspace/miniforge3/bin/conda clean -a -y --force-pkgs-dirs
CONDA_PKGS_DIRS=/workspace/miniforge3/pkgs /workspace/miniforge3/bin/conda clean -a -y --force-pkgs-dirs
```

## Stage 1 data

```bash
# 칩 위치 목록(헤더만), 창 선택, 창 안 PS-RGB·MS·GeoJSON 다운로드, G1
/workspace/miniforge3/envs/geo/bin/python run.py data
```

## Stage 2 raster (G2~G7, G8은 Stage 5 sr_run에서)

```bash
/workspace/miniforge3/envs/geo/bin/python run.py raster
```

## 스모크 Stage 1~6 (configs/default.yaml smoke: true, 2026-10-29 성공)

```bash
# 결과는 data/smoke, runs/smoke, results/. 각 단계 소요 시간·GPU·RAM은 logs/timing.csv
P=/workspace/miniforge3/envs/geo/bin/python
T=$(date +%Y%m%d_%H%M); nohup $P run.py data raster > logs/smoke_data_raster_$T.log 2>&1 &
$P run.py tiles_labels > logs/smoke_tiles_labels_$(date +%Y%m%d_%H%M).log 2>&1
$P run.py train_det > logs/smoke_train_det_$(date +%Y%m%d_%H%M).log 2>&1
$P run.py sr > logs/smoke_sr_$(date +%Y%m%d_%H%M).log 2>&1
$P run.py eval > logs/smoke_eval_$(date +%Y%m%d_%H%M).log 2>&1
PROJ_DATA=/workspace/miniforge3/envs/geo/share/proj $P -m pytest -q tests
```

## 전체 실행 (configs/default.yaml smoke: false, 2026-10-30 성공)

```bash
# 결과는 data/full, runs/full, results/. 탐지기 학습 후 정지 지점에서 사용자 확인.
P=/workspace/miniforge3/envs/geo/bin/python
T=$(date +%Y%m%d_%H%M%S); nohup $P run.py data raster tiles_labels train_det det_calib > logs/full_$T.log 2>&1 &
# max_det_min 1000 → 3000 변경 후(사용자 결정): label_stats.json만 갱신, D_LR 200에폭 민감도 학습(det_sens, 커밋 da7fe24), val 보정 재실행
T=$(date +%Y%m%d_%H%M%S); nohup bash -c "set -e; PROJ_DATA=/workspace/miniforge3/envs/geo/share/proj $P src/labels.py && $P run.py det_sens det_calib" > logs/det_sens_calib_$T.log 2>&1 &
# D_LR 민감도 판정(+0.0124 ≥ 0.01)에 따라 epochs 200으로 재학습(100에폭 결과는 runs/full/D_*_e100으로 mv). 민감도 코드는 커밋 da7fe24.
mv runs/full/D_HR runs/full/D_HR_e100 && mv runs/full/D_LR runs/full/D_LR_e100
T=$(date +%Y%m%d_%H%M%S); nohup $P run.py train_det det_calib sr > logs/full_e200_sr_$T.log 2>&1 &
# SR 단계 후 test 평가 직전 docs/frozen_eval.md 기록·커밋, 그다음 test 평가
cp data/full/s6_finetune.json results/s6_finetune.json
T=$(date +%Y%m%d_%H%M%S); nohup $P run.py eval > logs/full_eval_$T.log 2>&1 &
```
