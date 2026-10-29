# 환경 보고서

## Stage 0 하드웨어 측정 (2026-10-29, 파드 jzyfcpm93plj55, EUR-IS-1)

원본 출력: logs/stage0_hw_20261029_0925.log. 이전 파드 측정(logs/stage0_hw.log, 54.4코어·overlay 디스크)은 이 파드에 해당하지 않는다.

| 항목 | 측정값 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| GPU | RTX 4090 24564MiB × 1, compute 8.9, 드라이버 595.91.07, 드라이버 CUDA 13.2 |
| CPU | `nproc` 64. cgroup cpu.max 1360000/100000 = 13.6 CPU, `RUNPOD_CPU_COUNT=16` |
| RAM | `free -h` 251Gi(호스트 표시). cgroup memory.max 61,999,996,928 B(≈62GB), `RUNPOD_MEM_GB=62` |
| /workspace | 네트워크 볼륨 `mfs#eur-is-1.runpod.net:9421[/networkvolumes/dyo07en3fi]`(FUSE). `df`는 클러스터 전체(2.1P)를 표시해 볼륨 할당량을 알 수 없다 |
| /root | 컨테이너 overlay 50G(사용 411M), 파드 재시작 시 사라진다 |

### 읽기 속도 (4GiB 파일, dd)

| 경로 | 순차 1MiB O_DIRECT | 4KiB O_DIRECT | 작은 파일 2000×64KiB 쓰기 / 읽기 |
|---|---|---|---|
| /workspace | 650 MB/s | 78 MB/s | 11.24 s / 4.40 s |
| /root | 3.4 GB/s | 152 MB/s | 0.18 s / 0.07 s |

### 측정값에 따른 판단
- CPU: gdal_threads=12, 프로세스 병렬 상한 n_proc=12(13.6코어 중 시스템 몫). GPU 1장이라 단계가 순서대로 돌아 학습 로더 워커 8개와 겹치지 않는다. 프로세스당 RAM 약 5GB.
- I/O: /root가 순차 약 5배, 작은 파일 약 60배 빠르다. 학습 직전 타일을 /root/tiles_cache로 복사한다.
- 설치: conda 환경은 규칙대로 /workspace/miniforge3에 두지만 작은 파일이 많아 설치·import가 느릴 수 있다.

### 시작 시점 /workspace 사용량 (du -sh)
| 경로 | 크기 | 소유 |
|---|---|---|
| /workspace/Vla_check | 33G | 다른 프로젝트(건드리지 않음) |
| /workspace/.cache (huggingface, pip, uv) | 29G | 다른 프로젝트로 보임(건드리지 않음) |
| /workspace/.venvs/openvla | 16G | 다른 프로젝트(건드리지 않음) |
| /workspace/rs-sr-detect | 9.0M | 이 프로젝트 |

## E3 검증 결과 (2026-10-29)

| 환경 | Python | torch | CUDA(torch 빌드) | `torch.cuda.is_available()` / 장치 | 담당 모델 최소 추론 | 로그 |
|---|---|---|---|---|---|---|
| geo | 3.12 | 2.15.0+cu132 | 13.2 | True / NVIDIA GeForce RTX 4090 | YOLO26s-OBB 로드, LPIPS | logs/stage0_verify_geo_20261029_1303.log |
| sr | 3.10.13 | 2.1.2+cu118 | 11.8 | True / NVIDIA GeForce RTX 4090 | S2 SwinIR 128→512, S3 RealESRNet 179→716, S4 RealESRGAN 179→716, S5 ResShift v3 256×512→1024×2048 | logs/stage0_verify_sr_20261029_1303.log |
| osediff | 3.10 | 2.0.1+cu117 | 11.7 | True / NVIDIA GeForce RTX 4090 | S7 OSEDiff 128→512 | logs/stage0_verify_osediff_20261029_1303.log |

- sr: `pip list` 기준 torch 2.1.2+cu118, torchvision 0.16.2+cu118, xformers 0.0.23.post1+cu118(torch 2.1.2용 빌드), triton 2.1.0. basicsr·gfpgan·Real-ESRGAN 설치 후에도 ResShift 고정 버전이 바뀌지 않았다.
- osediff: torch 2.0.1, torchvision 0.15.2, xformers 0.0.20, diffusers 0.25.0, transformers 4.28.1, huggingface_hub 0.25.0(OSEDiff requirements.txt 고정값).
- Ultralytics 8.4.192, YOLO 가중치 `yolo26s-obb.pt`(https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-obb.pt).
- 의존 모델 커밋 해시와 SHA256은 third_party/VERSIONS.md.

### 설치 중 발생한 문제와 조치
- SD2.1-base 다운로드 실패: 파드 템플릿 환경 변수 `HF_HUB_ENABLE_HF_TRANSFER=1` 때문에 huggingface_hub가 `hf_transfer`를 요구했다(osediff 환경에 없음). 패키지를 추가하지 않고 setup_envs.sh에서 `HF_HUB_ENABLE_HF_TRANSFER=0`으로 고정한 뒤, 가중치 단계만 다시 실행했다(logs/stage0_weights_20261029_1250.log).
- verify_env_sr.py: `realesrgan/__init__.py`의 `from .version import *`는 `__version__`을 내보내지 않아 `realesrgan.version`에서 읽도록 고쳤다. ResShift 샘플은 저장소에 실제 있는 `testdata/RealSet65/00003.png`로 바꾸고, 출력명은 공식 sampler 규칙(`{stem}.png`)을 따른다.
- `/home/notebook/data/group/LowLevelLLM/LLM/bert-base-uncased`(OSEDiff ram 코드의 하드코딩 경로) 심볼릭 링크는 컨테이너 디스크에 있어 파드 재시작 시 사라진다. setup_envs.sh를 다시 실행하면 복구된다.

## 설치 후 정리와 /workspace 사용량 (du -sh /workspace/*)

| 경로 | conda clean 전 (12:38) | conda clean·pip 캐시 삭제 후 + 가중치 다운로드 후 (12:58) |
|---|---|---|
| /workspace/cache | 16G | 5.0G |
| /workspace/miniforge3 | 33G | 33G (geo 12G, osediff 9.5G, sr 9.3G, base pkgs 2.3G) |
| /workspace/weights | 264M | 12G (osediff RAM 5.3G, sd21_base 4.9G, resshift 669M 등) |
| /workspace/rs-sr-detect | 919M | 933M |
| /workspace/Vla_check (다른 프로젝트) | 33G | 33G |

- `conda clean -a -y`는 tarball 179개(262.7MB)와 미사용 패키지 19개를 지웠고 pip 캐시는 삭제됐다.
- /workspace/cache/conda에는 압축 해제된 패키지 디렉터리 162개(5.0G)가 남았다. MooseFS에서는 하드링크가 만들어지지 않아(링크 수 1) 환경과 파일을 공유하지 않는 복사본이다. 지우려면 `conda clean -a -y --force-pkgs-dirs`가 필요하며 승인 대기 중이다.
- `df`는 클러스터 전체(2.1P, 63%)를 표시해 볼륨 할당량 대비 80% 판정에 쓸 수 없다.
