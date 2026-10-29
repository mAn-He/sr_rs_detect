# 역할
너는 원격탐사 컴퓨터비전 실험 저장소 `rs-sr-detect`를 만들고 실행하는 ML 엔지니어다.
이 문서는 프로젝트 전체 지시서다. 새 세션을 시작할 때마다 이 문서를 기준으로 현재 진행 상황(RUNBOOK.md, docs/, git log)을 확인하고 이어서 진행한다.

# 작업 환경
- 너는 RunPod 파드 안에서 실행 중이다. 코드 작성, 환경 설치, 데이터 다운로드, 학습, 추론, 평가를 모두 여기서 직접 실행한다.
  - OS: Ubuntu 22.04, NVIDIA RTX 4090(24GB) 1장, RAM 62GB(cgroup 한도), CPU 13.6코어(cgroup 할당량, RUNPOD_CPU_COUNT=16. `nproc`은 64를 표시하므로 쓰지 않는다)
  - 디스크: `/workspace`는 영구 네트워크 볼륨(MooseFS FUSE, 순차 읽기 약 650 MB/s, 작은 파일 접근이 느리다). `/root`는 파드 재시작 시 사라지는 컨테이너 디스크(overlay 50G, 순차 읽기 약 3.4 GB/s).
  - 저장소 위치: /workspace/rs-sr-detect
- /workspace만 파드를 재시작해도 남는다. 가상환경, 데이터, 가중치, 결과, 로그는 모두 /workspace 아래에 둔다.
- 경로는 하드코딩하지 않는다. configs/default.yaml의 `root: /workspace/rs-sr-detect` 하나에서 파생시킨다.
- 10분 이상 걸리는 작업은 `nohup ... > logs/<stage>_<time>.log 2>&1 &`로 백그라운드 실행하고 로그를 주기적으로 확인한다.
- 파괴적인 명령(rm -rf, 기존 환경 삭제, 데이터 재다운로드)은 실행 전에 나에게 묻는다.
- 단계마다 git commit 한다(데이터, 가중치, 대용량 결과는 .gitignore).
- RUNBOOK.md에는 네가 실제로 실행해 성공한 명령만 순서대로 기록한다. 새 파드에서 이 문서만 보고 재현할 수 있어야 한다.

# 목표의 우선순위
- 0순위: GDAL과 OpenCV를 실제로 필요한 곳에서 제대로 사용하는 위성영상 파이프라인
- 1순위: 범주별 대표 SR 방법 간 비교
- 2순위: SR이 다음 과제(건물 탐지)에 미치는 영향 측정

# 연구 질문
같은 시점에 같은 위성이 촬영한 실제 저해상도 영상(멀티스펙트럼, 약 1.2m)만 있을 때,
- 경로 1: 저해상도 그대로 건물을 탐지하는 것과
- 경로 2: SR로 0.3m로 복원한 뒤 탐지하는 것 중 무엇이 나은가?
- 경로 2에서 SR 방법별로 무엇이 다른가?
반드시 답할 하위 질문:
1. SR 화질 지표(PSNR/SSIM/LPIPS) 순위와 탐지 성능 순위가 같은가
2. 건물 면적(m²) 구간별 재현율이 어떻게 달라지는가
3. SR이 만든 거짓 양성(정답에 없는 건물 탐지)과 가짜 경계가 얼마나 되는가
4. SR로 늘어나는 처리 시간 대비 탐지 성능 이득은 얼마인가
- 합성 열화는 사용하지 않는다. 모든 저해상도 입력은 SpaceNet이 제공한 실제 멀티스펙트럼 GeoTIFF에서 만든다. 배율은 ×4 하나다.
- 이 실험은 "PAN이 없는 1.2m급 영상(다른 위성, PAN 미제공 자료)을 0.3m급으로 올려야 하는 상황"의 대리 실험이다. README 첫머리에 이 정의를 쓴다.

# 작업 원칙: 불필요한 코드 없이 작동할 것
1. 모든 라이브러리와 모델은 공식 저장소와 공식 문서를 먼저 읽고 그 사용법을 따른다. 문서와 이 지시가 충돌하면 멈추고 보고한다.
2. 공식 코드를 다시 구현하지 않는다. SR 모델과 탐지기는 공식 스크립트나 공식 Python API를 호출하고, 우리 코드는 입출력 변환(GeoTIFF ↔ 이미지, 좌표 정보 부여)만 담당한다.
3. 실행 경로에서 쓰이지 않는 함수, 파일, 클래스, 설정 키, CLI 인자를 만들지 않는다.
4. 추상화 계층, 레지스트리, 플러그인 구조, 범용 유틸 모음을 만들지 않는다. 한 파일은 한 단계의 일만 한다.
5. 미래 대비용 옵션이나 "혹시 몰라" 분기를 넣지 않는다. 이 문서에 적힌 기능만 구현한다.
6. 조용한 대체(fallback)와 예외 삼키기를 금지한다. 조건이 틀리면 assert나 명확한 에러로 멈춘다.
7. TODO, 플레이스홀더, mock, 더미 데이터를 최종 코드에 남기지 않는다.
8. 테스트는 격자 정렬, 픽셀↔지리 좌표 변환, OBB 라벨 변환에만 둔다. 작은 합성 배열로 검증한다.
9. 마무리 전에 `ruff check`(F401, F841 포함)와 `vulture`로 죽은 코드를 모두 제거하고 docs/code_check.md에 결과를 남긴다.
10. 수치를 지어내지 않는다. README의 모든 숫자는 results/의 CSV와 logs/에서 나온다.
11. 공식 가중치 URL이 깨져 있으면 다른 가중치로 몰래 바꾸지 말고 보고한다.

# 가상환경
목표: 모든 구성 요소가 함께 정상 작동하는 단일 환경을 먼저 시도하고, 안 되면 호환되는 묶음별로 환경을 나눈다.
E1. 요구사항 조사: 각 구성 요소의 공식 요구사항을 docs/env_matrix.md 표로 정리한다.
    (구성 요소 | Python | PyTorch | torchvision | CUDA | 고정 버전 요구 | 추가 의존성 | 출처 URL)
    대상: GDAL/rasterio/geopandas/shapely/OpenCV, Ultralytics, BasicSR, Real-ESRGAN, SwinIR, ResShift, OSEDiff, (활성화 시) AdcSR
    알려진 단서:
    - ResShift 공식 환경은 Python 3.10, PyTorch 2.1.2, xformers.
    - OSEDiff와 AdcSR은 공식 README 기준 Python 3.10과 각자의 requirements.txt를 쓴다.
    그 외 버전은 반드시 공식 문서로 확인하고 추측하지 않는다.
E2. 단일 환경 설계: 충돌이 없으면 `rsx` 하나로 설계한다. 명확한 충돌이 있으면 바로 분리안을 제시한다.
E3. 검증: `nvidia-smi`로 드라이버/CUDA를 확인해 맞는 PyTorch 빌드를 고른다.
    환경마다 scripts/verify_env_<name>.py를 실행한다: 모든 패키지 import, GPU 인식,
    담당 모델 각각의 공식 최소 추론을 샘플 1장으로 실행. 결과를 docs/env_report.md에 기록한다.
    하나라도 실패하거나, 해결하려면 다른 구성 요소의 고정 버전을 깨야 하면 E4로 간다.
E4. 분리 규칙: 호환되는 것끼리 묶고, 필요하면 모델별로 환경을 만든다.
    확정 구성(docs/env_matrix.md 근거): geo(GDAL·rasterio·geopandas·shapely·OpenCV·Ultralytics·평가), sr(BasicSR·Real-ESRGAN·SwinIR·ResShift),
    osediff. AdcSR 활성화 시 OSEDiff와 torch 고정 버전이 달라 별도 환경.
    환경 간에는 import로 연결하지 않고 디스크 파일(GeoTIFF, PNG, CSV, JSON)로만 주고받는다.
    단계→환경 매핑은 configs/default.yaml의 `envs:` 한 곳에 둔다. run.py는 geo 환경에서 실행하고,
    각 단계를 해당 환경의 python(/workspace/miniforge3/envs/<env>/bin/python)으로 subprocess 실행한다.
E5. 설치와 재현성: Miniforge를 /workspace/miniforge3에 설치하고 GDAL 계열은 conda-forge로 설치한다.
    환경마다 env/<name>.yml과 env/<name>.lock.txt(`conda list --explicit` + `pip freeze`)를 남긴다.
    Ultralytics 패키지 버전과 사용한 YOLO 가중치 파일명을 env_report.md에 고정 기록한다.
    scripts/setup_envs.sh는 여러 번 실행해도 안전해야 한다. 최종 구성과 이유를 env_report.md에 적는다.

# 저장 공간 최소화 (반드시 지킬 것)
- Vegas 중 서로 인접한 칩 1,200장 내외만 받는다. 먼저 칩 위치 목록만 내려받아 연속된 직사각형 영역을 고르고,
  그 영역의 PS-RGB, MS, 건물 GeoJSON만 다운로드한다. 고른 범위를 docs/data_report.md에 기록한다.
- 다른 AOI, PAN, PS-MS, tarball은 받지 않는다.
- 모자이크와 좌표계 변환은 VRT로만 만든다. 실제 래스터 파일은 최종 8비트 타일과 SR 결과만 쓴다.
- 모든 GeoTIFF는 `COMPRESS=DEFLATE, TILED=YES`로 저장한다.
- SR 결과는 test 블록에만 생성한다.
- 예외: 공식 SR 스크립트가 GeoTIFF를 읽지 못하면 임시 PNG를 컨테이너 디스크(/root/tmp_sr)에만 만들고, GeoTIFF 변환과 G8 좌표 검사가 끝나면 바로 삭제한다.
- 기본값: `enable_resshift: true`, `enable_osediff: true`, `enable_adcsr: false`. false면 해당 환경과 가중치를 만들지 않는다.
- 모델 가중치와 의존 모델(SD2.1-base, RAM, DAPE 등)은 /workspace/weights에 한 번만 받는다.
- CONDA_PKGS_DIRS와 PIP_CACHE_DIR를 /workspace/cache로 지정하고, 설치 직후 `conda clean -a -y`와 pip 캐시를 삭제한다.
- Stage마다 `du -sh /workspace/*`를 env_report.md에 기록하고, 사용량이 볼륨의 80%를 넘으면 멈추고 보고한다.

# 학습·처리 가속 (공식 옵션만 사용, 별도 분산 코드 작성 금지)
- Stage 0에서 `nproc`, `free -h`, `nvidia-smi`, `df -h`와 /workspace 읽기 속도를 측정해 env_report.md에 기록하고, 이 값으로 아래 설정의 기본값을 정한다.
- CPU 병렬:
  - gdalwarp는 `-multi -wo NUM_THREADS=<config의 gdal_threads, 12>`를 쓴다. ALL_CPUS는 cgroup 할당량이 아닌 64코어를 인식할 수 있어 쓰지 않는다.
  - 블록 단위 전처리(밴드 선택, 리샘플, 타일 저장)와 지리 평가 매칭, 부트스트랩, 엣지 지표 계산은 블록별로 프로세스 병렬 처리한다.
  - GDAL·타일링·평가의 프로세스 병렬 개수는 12개를 넘기지 않는다(13.6코어 중 시스템 몫을 남긴다. GPU 1장이라 단계가 순서대로 실행되므로 학습 로더 워커와 동시에 돌지 않는다. 프로세스당 RAM 약 5GB 이내).
  - 학습 데이터 로더 워커는 8개(Ultralytics 공식 기본값)로 한다.
  - 배치 크기: 24GB에서 메모리 부족이 나지 않는 최대값의 약 80%(config와 decisions.md에 값과 근거 기록).
  - YOLO 에폭 상한은 Ultralytics 공식 기본값, patience=20으로 조기 종료. 배치는 Ultralytics 공식 `batch=0.80`(GPU 메모리 80% AutoBatch)을 쓰고 실제 배치 값을 기록한다. D_HR, D_LR 각각 3시간을 넘을 것으로 예상되면 에폭을 줄이고 보고한다.
  - S6 반복 횟수는 공식 파인튜닝 설정에서 출발하되 학습 시간이 3시간을 넘지 않게 줄인다.
- RAM: 학습 데이터가 RAM 여유분 안에 들어가면 Ultralytics `cache` 옵션으로 이미지를 RAM에 올려 CPU 디코딩 부담을 줄인다.
  사용 RAM과 GPU 사용률을 timing.csv에 함께 기록한다.
- GPU:
  - GPU가 1장이면 작업을 순서대로 실행한다.
  - GPU가 2장 이상이면 서로 독립인 작업을 GPU마다 하나씩 동시에 실행한다(D_HR과 D_LR 학습, SR 방법별 추론).
    한 학습을 여러 GPU로 쪼개는 DDP는 쓰지 않는다.
  - Ultralytics AMP와 SR 추론 fp16은 공식 기본값을 유지한다.
- I/O: 학습 직전에 해당 학습의 타일만 컨테이너 디스크(/root/tiles_cache)로 복사해 읽고, 학습이 끝나면 삭제한다.
  Stage 0 측정에서 /root가 /workspace보다 순차 읽기 약 5배, 작은 파일 접근 약 60배 빨라 복사를 기본으로 한다.
- 스모크 테스트에서 GPU 사용률과 단계별 소요 시간을 logs/timing.csv에 기록하고, 병목(CPU, I/O, GPU)을 보고한다.

# 먼저 읽을 공식 문서
- GDAL: https://gdal.org/en/stable/programs/ (gdalinfo, gdalbuildvrt, gdalwarp, gdal_translate, VRT·COG 드라이버)
- rasterio 문서 (windowed read/write, transform, Affine)
- OpenCV 문서 (phaseCorrelate, minAreaRect, boxPoints, dnn.NMSBoxesRotated, Canny, Laplacian, dilate)
- Ultralytics OBB: https://docs.ultralytics.com/tasks/obb/ , https://docs.ultralytics.com/datasets/obb/
  - 라벨 형식: `class_index x1 y1 x2 y2 x3 y3 x4 y4` (0~1 정규화)
- SwinIR: https://github.com/JingyunLiang/SwinIR
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN (docs/model_zoo.md, docs/Training.md의 paired data 파인튜닝)
- BasicSR: https://github.com/XPixelGroup/BasicSR
- ResShift: https://github.com/zsyOAOA/ResShift
- OSEDiff: https://github.com/cswry/OSEDiff
- (활성화 시) AdcSR: https://github.com/Guaishou74851/AdcSR
- SpaceNet: https://registry.opendata.aws/spacenet/
- 선행 연구(README에서 인용): GeoSR-Bench, https://arxiv.org/abs/2605.00310
- 클론한 외부 저장소는 third_party/에 두고 커밋 해시를 third_party/VERSIONS.md에 기록한다.

# 데이터
- SpaceNet 2, AOI_2_Vegas (`aws s3 ... --no-sign-request`)
  - 고해상도 정답: s3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_2_Vegas/PS-RGB/
  - 실제 저해상도 입력: 같은 상위 경로의 멀티스펙트럼(MS) 폴더. `aws s3 ls`로 정확한 폴더명을 확인한다.
  - 라벨: s3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_2_Vegas/geojson_buildings/
- 라이선스 CC BY-SA 4.0, README에 출처 표기.
- PS-RGB는 PAN과 MS를 합친 팬샤프닝 산출물이다. README의 데이터와 한계 섹션에 적는다.
- 실제 좌표계, 해상도, dtype, 비트 심도, 밴드 수와 순서는 G1에서 확인해 기록한다. 추측하지 않는다.

# 저장소 구조 (이 범위를 넘는 파일은 만들지 않는다)
rs-sr-detect/
  CLAUDE.md  RUNBOOK.md  README.md  run.py  .gitignore
  configs/default.yaml
  env/        # <name>.yml, <name>.lock.txt
  scripts/    # setup_envs.sh, download_data.sh, verify_env_<name>.py
  src/
    gdal_ops.py      # G1~G8
    tiling.py        # G9
    labels.py        # G10
    sr_run.py        # S0~S8 공식 스크립트 호출과 결과 GeoTIFF 저장
    sr_finetune.py   # S6
    det_train.py  det_infer.py
    merge.py         # G11
    geo_eval.py  sr_metrics.py  report.py
  tests/test_geometry.py
  docs/            # env_matrix.md, env_report.md, data_report.md, code_check.md, decisions.md
- 구현 중 필요 없는 파일은 만들지 않는다. 더 필요하면 이유를 보고한다.

# GDAL 처리 명세 (이 순서와 방식을 지킬 것)
G1 점검: PS-RGB와 MS 각각 샘플 칩 5장에 `gdalinfo -stats`. CRS, 픽셀 크기, dtype, 비트 심도, NoData, 밴드 수,
    밴드 설명(있으면)을 data_report.md에 기록한다. MS 픽셀 크기가 PS-RGB의 약 4배가 아니면 멈추고 보고한다.
G2 모자이크: `gdalbuildvrt`로 PS-RGB 칩은 hr_src.vrt, MS 칩은 ms_src.vrt로 묶는다.
G3 기준 격자: 지리좌표계(도 단위)면 `gdalwarp -of VRT -t_srs EPSG:32611`로 UTM 11N에 맞춘다.
    - HR: `-tr 0.3 0.3 -r cubic` → hr_utm.vrt
    - LR: MS에서 R, G, B 밴드를 골라(`-b`로 지정, 밴드 번호는 G1에서 확인한 근거와 함께 기록) `-tr 1.2 1.2 -r cubic` → lr_utm.vrt
    - 두 격자의 출력 범위(-te)를 같은 1.2m 정수배로 맞춰 LR 1픽셀이 HR 4×4픽셀에 정확히 대응하게 한다.
G4 공간 분할: 576m×576m 블록(= LR 480px = HR 1920px) 단위로 train/val/test = 70/15/15 배정(splits.geojson). 무작위 타일 분할 금지.
    블록 격자는 공통 범위의 좌상단에서 시작하고, 가장자리에서 576m를 채우지 못하는 블록은 버리고 버린 면적(km²)을 data_report.md에 기록한다.
G5 정합 검사: 블록마다 HR을 1.2m로 평균 축소한 영상과 LR의 밝기 채널로 OpenCV `phaseCorrelate`를 계산해
    블록별 어긋남(LR 픽셀 단위)을 data_report.md에 기록한다. 0.5 LR 픽셀을 넘는 블록이 있으면 멈추고 보고한다. 보정은 하지 않는다.
G6 8비트 변환: HR과 LR은 서로 다른 제품이므로 각각 train 블록에서 밴드별 2~98 백분위를 따로 계산해
    `gdal_translate -ot Byte`의 밴드별 scale 옵션으로 적용한다(옵션명은 공식 문서 확인, 결과는 VRT).
    같은 장면의 HR과 LR 색 비교 PNG를 남기고, 색 차이가 크면 보고한다.
G7 보간 SR(test 블록만): `gdalwarp lr_8bit.vrt -te <test 블록 범위> -tr 0.3 0.3 -r cubic` 및 `-r lanczos`.
G8 딥러닝 SR 저장: rasterio로 저장, transform = LR 타일 transform × Affine.scale(1/4).
    저장 후 같은 지리 창의 HR 타일과 transform·shape가 같은지 assert.
G9 타일링: 지리 창 크기 307.2m(= HR 1024px = LR 256px), 겹침 LR 32px(= HR 128px, 12.5%), 이동 간격 LR 224px(= HR 896px, 268.8m).
    타일은 576m 블록 안에서만 만든다. 블록당 2×2 타일, 블록 내 LR 시작점 0과 224(마지막 타일이 블록 끝 480에 정확히 맞는다). 모든 타일 시작점이 1.2m 격자에 맞는지 assert. 창 목록은 tiles.geojson 하나로 공유한다.
    HR·LR 타일은 모든 분할, SR 타일은 test 분할만 만든다(S6 학습용 쌍은 train 블록 HR·LR 타일에서 바로 읽는다).
G10 라벨: 건물 GeoJSON → UTM → 타일 창으로 clip(원래 면적의 50% 미만이면 제외) → 픽셀 좌표(~transform)
    → OpenCV minAreaRect/boxPoints → 정규화 → YOLO OBB txt. HR(1024)용과 LR(256)용 라벨을 모두 만든다.
    건물별 면적(m²)과 partialBuilding 속성은 CSV로 보관.
G11 결과 병합: 타일 예측 → 전역 픽셀 좌표 → `cv2.dnn.NMSBoxesRotated` → UTM 폴리곤 → EPSG:4326 GeoJSON.
G12 지리 평가: test 블록에서 UTM 회전 IoU ≥ 0.5로 매칭.
    AP50, 면적 구간별 재현율(기본 <50 / 50–150 / 150–400 / >400 m², config로 조정), 거짓 양성 수.
    Ultralytics val 지표(타일 단위)도 기록하되 결론은 지리 평가 기준으로 낸다.
G13 시각화: 대표 장면을 `gdal_translate -of COG`로 저장해 예측 GeoJSON과 함께 QGIS에서 열 수 있게 한다.

# 탐지기 (YOLO 하나로 고정)
- 탐지기는 Ultralytics 공식 문서 기준 최신 OBB 모델(작성 시점 YOLO26-OBB) 하나만 쓴다. 다른 탐지기 구조는 사용하지 않는다.
- D_HR: s 크기, 공식 DOTA 사전학습 OBB 가중치에서 시작, HR 1024 타일, imgsz=1024
- D_LR: 같은 구조, 같은 에폭 예산, 같은 시드, LR 256 타일, imgsz=256 (경로 1)
- val 블록으로 최적 가중치를 고르고 test 블록에서만 최종 평가한다.
- 탐지기는 고정한다. SR 방법마다 탐지기를 다시 학습하지 않는다.
  결론은 항상 "고해상도로 학습한 탐지기를 그대로 쓸 때"라는 조건을 붙여 쓴다.

# SR 방법 (ID 고정, 입력은 모두 같은 8비트 LR 타일, 배율은 모두 ×4)
- 보간: S0 gdal_cubic, S1 gdal_lanczos (G7)
- 회귀·화질 지향: S2 swinir_classical_x4 (공식 DF2K classical ×4), S3 realesrnet_x4plus (Real-ESRGAN 공식, GAN 없이 학습된 같은 구조)
- GAN: S4 realesrgan_x4plus (Real-ESRGAN 공식)
- Diffusion 효율형: S5 resshift_v3_x4 (공식 inference_resshift.py, --task realsr --scale 4 --version v3. 라이선스 NTU S-Lab License 1.0)
- 도메인 파인튜닝: S6 realesrgan_ft_spacenet (공식 Training.md의 paired data 파인튜닝. 공식 finetune_realesrgan_x4plus_pairdata.yml을 복사해 pretrain_network_g만 RealESRGAN_x4plus.pth로 바꾸고 netD는 RealESRGAN_x4plus_netD.pth. S6−S4 = 파인튜닝 효과, train 블록의 실제 MS–PS 타일 쌍만, 반복 횟수는 config로 고정)
- Diffusion 1스텝: S7 osediff_x4 (공식 test_osediff.py. 의존 모델 SD2.1-base, RAM, DAPE는 공식 README의 링크에서만 받는다. SD2.1-base는 README가 링크한 미러 Manojb/stable-diffusion-2-1-base를 쓰고 미러 이름·커밋 해시·SHA256을 third_party/VERSIONS.md에 기록, README 한계에 "원본과 동일성 미검증" 명시. 코드는 Apache-2.0, 베이스 모델 라이선스는 따로 확인해 README에 기록)
- Diffusion 1스텝 경량: S8 adcsr_x4 (enable_adcsr=true일 때만)
- 애니메이션용 모델(RealESRGAN_x4plus_anime_6B 등)은 도메인이 달라 사용하지 않는다.
- 큰 영역은 각 모델의 공식 tile 옵션 또는 G9 타일 단위로 처리한다.

# 판정 기준 (SR은 다음 과제에 미치는 영향으로 판정한다)
- 주 지표는 탐지 성능(AP50_geo, 면적 구간별 재현율)이다. PSNR/SSIM/LPIPS와 OpenCV 엣지 지표는 결과를 설명하는 보조 지표로만 쓴다.
- 기준점: 경로 1(D_LR on LR)을 하한, HR 원본(D_HR on HR)을 상한으로 둔다.
- 각 SR 방법마다 계산한다:
  - gain = AP50(경로 2, 해당 SR) − AP50(경로 1), 부트스트랩 95% 신뢰구간 포함
  - recovery = gain ÷ (AP50(HR) − AP50(경로 1))
  - verdict = gain 신뢰구간 하한 > 0 이면 "useful", 상한 < 0 이면 "harmful", 그 외 "inconclusive"
  - 면적 구간별 재현율에도 같은 gain과 verdict를 계산한다.
- README의 결론은 verdict 열을 근거로 쓴다. 화질 지표 순위로 결론을 내리지 않는다.

# 결과의 신뢰도와 판단 기록
- 신뢰구간: test 블록 단위 부트스트랩(1,000회, 시드 고정)으로 AP50, 면적 구간별 재현율, gain의 95% 신뢰구간을 계산한다.
  두 방법의 신뢰구간이 겹치면 README에서 "차이가 있다"고 단정하지 않는다.
- 판단 기록: docs/decisions.md에 주요 설계 선택마다 '선택 / 대안 / 이유 / 결과가 지지했는가'를 기록한다.
  최소 포함: 실제 MS를 저해상도로 쓴 이유, RGB 밴드 선택 근거, 576m 블록 크기, 307.2m 타일 크기,
  제품별 8비트 변환, 탐지기를 YOLO 하나로 고정한 이유와 그 한계, imgsz, SR 방법 선정과 제외 이유, 평가 IoU 기준.
  Stage 7에서 마지막 칸을 결과로 채운다.

# 성능 비교 결과 → results/summary.csv
- 행: HR 원본(상한선), 경로 1(D_LR on LR), 경로 2의 활성화된 S0~S8(D_HR on SR)
- 열: method, category, path, PSNR, SSIM, LPIPS, edge_precision, edge_recall, sharpness_ratio,
  AP50_geo, AP50_ci_low, AP50_ci_high, recall_<면적구간>(+각 신뢰구간),
  gain, gain_ci_low, gain_ci_high, recovery, verdict, false_positives, sr_sec_per_km2, det_sec_per_km2
- OpenCV 지표 정의(파라미터는 config에 고정):
  - Canny 임계값: HR train 타일 밝기 채널의 중앙값 기준으로 한 번 정하고 모든 방법에 같은 값을 쓴다(config와 decisions.md에 값과 근거 기록).
  - edge_precision: HR과 SR의 밝기 채널에 같은 파라미터로 cv2.Canny를 적용하고,
    SR 엣지 중 HR 엣지를 1픽셀 cv2.dilate한 영역 안에 있는 비율. 낮으면 없는 경계를 만든 것이다.
  - edge_recall: HR 엣지 중 SR 엣지를 1픽셀 dilate한 영역 안에 있는 비율.
  - sharpness_ratio: cv2.Laplacian 분산의 SR/HR 비율.
- 화질 지표 순위와 AP50 순위의 Spearman 상관을 계산해 README에 보고한다.
- 그림: (a) PSNR vs AP50 산점도(신뢰구간 오차막대), (b) 면적 구간별 재현율 막대그래프,
  (c) 같은 장면의 방법별 비교 그리드, (d) 거짓 양성 사례 확대, (e) km²당 처리 시간 vs AP50,
  (f) sharpness_ratio vs AP50

# 단계와 검증 체크
Stage 0 env: 하드웨어 측정(가속 섹션), E1~E5. ✔ 모든 verify 스크립트 통과, env_report.md 작성.
Stage 1 data: 영역 선택, PS-RGB·MS·GeoJSON 다운로드, G1. ✔ data_report.md 생성, 사용 용량 기록.
Stage 2 raster: G2~G8. ✔ LR 격자 × 4 = HR 격자(assert). ✔ G5 정합 검사 통과.
    ✔ HR/LR/S0 비교 PNG 3장, HR·LR 색 비교 PNG.
Stage 3 tiles_labels: G9~G10. ✔ 라벨 오버레이 PNG 10장(HR 5, LR 5).
    ✔ data_report.md에 건물 면적 분포, HR·LR 건물당 픽셀 수 분포, 분할별 건물 수 추가.
Stage 4 train_det: D_HR, D_LR. ✔ 학습 곡선과 val 지표 저장.
Stage 5 sr: 활성화된 S0~S8(S6는 파인튜닝 후 추론). ✔ 모든 결과가 HR 격자와 일치(assert).
    ✔ SR 화질 지표와 OpenCV 엣지 지표 CSV.
Stage 6 eval: 모든 행 탐지 → G11 → G12 → 부트스트랩 신뢰구간 → 판정 기준 계산 → summary.csv와 그림.
Stage 7 report: README.md에 목적과 대리 실험 정의, 데이터, GDAL·OpenCV 처리 흐름, 환경 구성과 이유,
    가속 설정과 병목 분석, 방법, 결과표(summary.csv에서 자동 생성), 해석, 선행 연구(GeoSR-Bench)와의 관계, 한계.
    docs/decisions.md 마지막 칸 완성.
    한계에 반드시 포함:
    - 정답 HR이 팬샤프닝 산출물이라는 점과 HR·LR의 제품별 색 차이
    - 탐지기를 YOLO 하나로 고정해, 탐지기가 팬샤프닝 질감에 익숙해서 생긴 차이를 분리하지 못한 점
    - 단일 도시, 데이터 일부만 사용, 건물 단일 클래스, 배율 ×4 하나
    - 각 SR 모델의 학습 열화와 실제 MS 특성의 차이
    - 제외한 방법과 이유, 데이터·모델 라이선스
    - SD2.1-base를 OSEDiff README가 링크한 개인 미러에서 받아 stabilityai 원본과 동일성을 검증하지 못한 점

# 스모크 테스트
- config의 `smoke: true`면 칩 20장(576m 블록이 3개 이상 나오게 고르고 train/val/test 각각 최소 1블록을 강제, 이 규칙은 smoke일 때만), 에폭 1, S6 반복 횟수 최소, 부트스트랩 50회로 Stage 1~6이 끝까지 돌아야 한다.
  활성화된 diffusion 모델(S5, S7)도 타일 몇 장으로 실제 추론을 거쳐야 한다.
- 스모크 테스트를 통과하면 결과, timing.csv의 병목 분석, 용량 사용량, G1·G5 확인 결과를 요약 보고하고 멈춘다.
  전체 실행은 내가 지시한 뒤 시작한다.

# 진행 방식
- 각 Stage를 마칠 때마다 검증 체크 결과, 사용 용량, 소요 시간을 짧게 보고한다.
- 검증 체크가 실패하면 원인과 해결안을 보고하고 내 확인을 받은 뒤 진행한다.

# 처음 할 일
1. 공식 문서를 읽고 docs/env_matrix.md를 작성한다. 이 지시와 충돌하거나 불확실한 점을 목록으로 보고한다.
2. Stage 0을 실행한다(하드웨어 측정, 단일 환경 시도, 필요 시 분리).
3. 저장소 뼈대와 configs/default.yaml을 만들고 정적 검사(ruff, vulture)를 통과시킨다.
4. smoke: true로 Stage 1~6을 실행해 통과시킨 뒤 보고하고 멈춘다.
