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
