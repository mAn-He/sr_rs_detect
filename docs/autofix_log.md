# 자동 수정 기록

형식: 증상 / 원인 / 조치 / 영향. 패키지·경로·코드 버그를 고칠 때마다 추가하고 커밋한다.
Stage 0 항목은 2026-10-29에 소급 기록했다.

## Stage 0

### 1. SD2.1-base 다운로드 실패 (setup_envs.sh)
- 증상: setup_envs.sh가 가중치 단계에서 `ValueError: Fast download using 'hf_transfer' is enabled (HF_HUB_ENABLE_HF_TRANSFER=1) but 'hf_transfer' package is not available` 로 멈췄다(logs/stage0_setup_20261029_1056.log).
- 원인: RunPod 파드 템플릿이 `HF_HUB_ENABLE_HF_TRANSFER=1`을 설정하지만 osediff 환경에는 `hf_transfer`가 없다.
- 조치: 패키지를 추가하지 않고 setup_envs.sh에서 `export HF_HUB_ENABLE_HF_TRANSFER=0`으로 고정. 가중치 단계만 다시 실행해 성공(logs/stage0_weights_20261029_1250.log).
- 영향: 다운로드 속도만 달라지고 받은 파일은 같다(VERSIONS.md의 커밋 해시·SHA256). 수정본 setup_envs.sh 전체 재실행은 미검증.

### 2. verify_env_sr.py: realesrgan 버전 출력 실패
- 증상: `AttributeError: module 'realesrgan' has no attribute '__version__'`
- 원인: `realesrgan/__init__.py`의 `from .version import *`는 밑줄로 시작하는 `__version__`을 내보내지 않는다(공식 저장소 동작).
- 조치: `import realesrgan.version`으로 바꾸고 그 모듈에서 버전을 읽는다.
- 영향: 검증 출력만 바뀐다. 추론 경로 영향 없음.

### 3. verify_env_sr.py: ResShift 샘플 경로 없음
- 증상: ResShift `sampler.py`의 `assert in_path.exists()` 실패.
- 원인: 스크립트가 지정한 `testdata/RealSet65/0801_pch_00001.png`가 ResShift 저장소(커밋 bb03b7d)에 없다.
- 조치: 저장소에 있는 `testdata/RealSet65/00003.png`를 쓰고, 출력명은 공식 sampler 규칙(`{stem}.png`)으로 확인한다.
- 영향: 검증 샘플만 바뀐다. 추론 명령 인자(`--task realsr --scale 4 --version v3`)는 그대로.

### 4. ruff SIM115 (verify_env_geo/sr/osediff.py)
- 증상: `ruff check scripts/`에서 SIM115(`open()`을 context manager 없이 사용) 3건.
- 원인: `yaml.safe_load(open(...))`로 설정 파일을 읽었다.
- 조치: `yaml.safe_load(Path(...).read_text())`로 변경. 이후 ruff·vulture 통과, 세 verify 재실행 통과(logs/stage0_verify_*_20261029_1303.log).
- 영향: 동작 같음.

### 5. osediff.yml: 공식 requirements.txt에 없는 torchvision·numpy 고정
- 증상: OSEDiff requirements.txt는 torchvision과 numpy를 고정하지 않는데, 그대로 설치하면 버전이 맞지 않는 패키지가 들어올 수 있다.
- 원인:
  - numpy: torch 2.0.1 휠은 NumPy 1.x로 빌드되어 NumPy 2.x에서는 torch의 numpy 연동이 깨진다. 고정이 없으면 pip가 최신 NumPy 2.x를 고를 수 있다.
  - torchvision: requirements.txt의 `basicsr`가 버전 없이 torchvision을 요구해, 고정이 없으면 pip가 최신 torchvision과 그에 맞는 새 torch를 설치해 공식 고정값 torch==2.0.1을 깰 수 있다. BasicSR 1.4.2는 torchvision<0.17도 필요하다(env_matrix.md).
- 조치: torch 2.0.1의 공식 짝인 `torchvision==0.15.2`, NumPy 1.x 최신 `numpy==1.26.4`로 고정. 공식 requirements.txt의 고정값은 하나도 바꾸지 않았다.
- 영향: 설치 결과 torch 2.0.1+cu117, torchvision 0.15.2, numpy 1.26.4(env/osediff.lock.txt). OSEDiff 최소 추론 통과.

## Stage 1

### 6. aws CLI 없음 (geo 환경)
- 증상: `aws` 명령이 시스템과 geo 환경 어디에도 없다.
- 원인: Stage 0 환경 설계에 S3 다운로드 도구가 빠져 있었다.
- 조치: `conda install -n geo -c conda-forge awscli`(2.38.3), env/geo.yml에 `awscli` 추가, env/geo.lock.txt 갱신. 설치 중 python-dateutil 3.0.0→2.9.0, urllib3 2.8.0→2.6.3 다운그레이드(pandas·requests 하한 조건 안). 패키지 캐시는 `conda clean -a -y --force-pkgs-dirs`로 삭제.
- 영향: verify_env_geo.py 재실행 통과(logs/stage1_verify_geo_20261029_1332.log).

### 7. download_data.sh: 칩 헤더 병렬 수집 출력 섞임
- 증상: `json.decoder.JSONDecodeError: Extra data`(logs/stage1_index_*.log 첫 실행).
- 원인: `xargs -P`로 병렬 실행한 `printf`와 `gdalinfo -json` 출력이 한 stdout에서 줄 단위로 섞였다.
- 조치: GDAL Python 바인딩(`gdal.Open('/vsis3/...').GetGeoTransform()`)을 ThreadPoolExecutor(n_proc=12)로 호출하고 한 프로세스에서 CSV를 쓴다. `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`로 칩당 헤더 읽기 약 9 s → 1 s.
- 영향: 3,850장 위치 목록 4분 16초, 파일은 받지 않았다.

### 8. download_data.sh: aws CLI 파일별 호출이 느림
- 증상: 파일별 `aws s3 cp`(12개 병렬)가 3분에 71장(합계 약 1 MB/s). 같은 파일을 curl로 받으면 1.1 s, aws CLI는 8.7~14.6 s.
- 원인: aws CLI(Python)가 FUSE 볼륨(/workspace/miniforge3)에서 수천 개 모듈을 import하는 시작 시간. IMDS 비활성화로는 일부만 줄었다.
- 조치: aws 한 프로세스에 `--recursive --exclude '*' --include '*_img<ID>.<ext>' ...` 목록을 넘겨 CLI 자체 동시 전송을 쓴다. 중단 시 남은 임시 파일 3개(`*.tif.<난수>`, 미완성)는 삭제했고 완성 파일은 크기(2,537,990 B)로 확인했다.
- 영향: 받는 파일 목록과 내용은 같다. 마지막에 폴더별 파일 수 = 창 칩 수를 assert.

### 9. osgeo 단독 import 시 EPSG 조회 실패 (PROJ 데이터 경로)
- 증상: `from osgeo import osr; SpatialReference().ImportFromEPSG(32611)`이 6(실패)을 반환하고 `PROJ: proj_create_from_database: Open of .../envs/geo/share/proj failed`. `rasterio`를 먼저 import하면 성공하고, CLI(gdalwarp, projinfo)는 정상.
- 원인: run.py가 conda 활성화 없이 환경의 python을 직접 실행해 `PROJ_DATA`가 설정되지 않았다. rasterio는 import 시 PROJ 경로를 스스로 지정한다.
- 조치: run.py가 각 단계 subprocess에 `PROJ_DATA=<환경>/share/proj`를 넘긴다. 설정 후 osgeo 단독 import에서도 EPSG 조회 성공.
- 영향: 지금까지의 G1~G5는 CLI 또는 rasterio를 먼저 import한 프로세스에서 실행돼 결과에 영향이 없다(로그에 PROJ 오류 없음).

### 10. G5 phaseCorrelate 홀수 DFT 크기 +0.5 px 편향
- 증상: G5에서 110/168 블록이 0.5 LR px 초과, 동서 dx가 모든 블록에서 +0.4~0.6.
- 원인: cv2.phaseCorrelate는 입력을 getOptimalDFTSize로 늘리는데 그 크기가 홀수인 축에서 정확히 +0.5 px 편향이 생긴다(동일 영상: 폭 128/144/150/160 → 0.000, 129/130/135 → 0.500). G5 창은 동서 크기가 대부분 홀수 DFT였다.
- 조치: 창을 n 이하 짝수 최적 DFT 크기로 자르고(even_dft), 같은 창에서 ECC 평행이동을 함께 기록한다(ecc_dx, ecc_dy, ecc_shift). 합성 테스트에서 ECC 오차 ≤ 0.017 px.
- 영향: 같은 데이터에서 0.5 초과 블록 110 → 9(ECC 7). 판정 임계값은 바꾸지 않았다.

## Stage 5 (스모크)

### 11. S6 파인튜닝이 가중치를 바꾸지 않음
- 증상: 스모크에서 S6 결과 지표가 S4와 소수점 끝자리까지 같았다. net_g_latest.pth의 params·params_ema가 RealESRGAN_x4plus.pth와 최대 차이 0.0.
- 원인: 스모크 train 쌍이 4개인데 공식 batch_size_per_gpu는 12이고 BasicSR 학습 로더는 drop_last=True라, 에폭당 배치가 0개였다. 로그는 "iters: 20"을 찍지만 실제 갱신은 없었다.
- 조치: smoke일 때만 공식 yml 옵션 dataset_enlarge_ratio를 ceil(12 / 쌍 수)로 올려 배치 1개 이상을 만든다. 전체 실행(쌍 수 ≥ 12)은 공식 값 1 그대로다. 학습 후 가중치가 바뀌었는지 assert한다.
- 영향: 스모크 S6만 다시 실행. 학습 설정(배치, 손실, 학습률)은 바꾸지 않았다.

## Stage 6 (스모크)

### 12. det_infer.py: 타일 단위 val의 data YAML에 train 키 없음
- 증상: `SyntaxError: .../tiles/hr/test.yaml 'train:' key missing`(Ultralytics check_det_dataset).
- 원인: Ultralytics는 split과 무관하게 data YAML에 train·val 키를 요구한다.
- 조치: test 평가용 YAML에 train·val도 images/test로 적는다(split='test'라 test만 읽는다).
- 영향: 없음. 타일 단위 지표만 기록한다.

### 13. det_infer.py: 첫 행(HR) 탐지 시간에 CUDA 초기화 포함
- 증상: 스모크 det_timing에서 HR 15.9 s, 나머지 0.2~0.5 s.
- 원인: 프로세스의 첫 predict 호출이 CUDA 초기화·모델 준비를 포함했다.
- 조치: 행마다 타일 1장으로 예열한 뒤 시간을 잰다. 재실행 후 HR 0.26 s.
- 영향: det_sec_per_km2만 바뀐다. 예측 결과는 같다.
