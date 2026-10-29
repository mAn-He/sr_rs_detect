# 설계 결정 기록

형식: 선택 / 대안 / 이유 / 결과가 지지했는가(Stage 7에서 채움)

## 타일 겹침과 이동 간격
- 선택: 창 307.2m(HR 1024px, LR 256px), 겹침 LR 32px(HR 128px, 12.5%), 이동 간격 LR 224px(HR 896px, 268.8m)
- 대안: 겹침 15%(이동 간격 217.6 LR px), 반올림한 218 LR px
- 이유: 15%는 이동 간격이 정수 LR 픽셀이 아니라 1.2m 격자에 맞지 않는다. 32px 겹침과 224px 간격은 LR·HR 모두 정수 픽셀이고 32의 배수다.
- 결과가 지지했는가:

## 타일과 블록 경계
- 선택: 블록 576m(LR 480px, HR 1920px). 타일은 블록 안에서만 2×2(LR 시작점 0, 224), 겹침 32px. 가장자리에서 576m를 채우지 못하는 블록은 버리고 버린 면적을 기록한다. 모든 타일 시작점이 1.2m 격자에 맞는지 assert한다.
- 대안: 600m 블록(LR 500px)에 시작점 0, 224, 244(마지막 두 타일이 92% 겹치고 블록당 9장), 블록 경계를 넘는 전역 격자 타일링
- 이유: 블록 경계를 넘는 타일은 분할 간 데이터 누수를 만든다. 600m에서는 블록 끝 맞춤 타일이 거의 중복된다. 576m = 224 + 256 LR px라서 2×2 타일이 32px 겹침으로 블록을 정확히 덮는다.
- 결과가 지지했는가:

## 스모크 테스트 분할
- 선택: smoke: true일 때만 576m 블록이 3개 이상 나오게 칩 20장을 고르고 train/val/test 각각 최소 1블록을 강제한다.
- 대안: 70/15/15 비율을 그대로 적용
- 이유: 칩 20장(약 0.8km²)에서 비율대로 나누면 val이나 test가 빌 수 있다.
- 결과가 지지했는가:

## 학습 예산
- 선택: YOLO epochs=100(공식 기본값), patience=20, batch=0.80(공식 AutoBatch, GPU 메모리 80%), workers=8, D_HR·D_LR 각각 3시간 상한. S6은 공식 paired data 파인튜닝 설정에서 출발해 3시간 이내로 반복 횟수를 줄인다. 실제 배치 값과 S6 반복 횟수는 Stage 4/5 측정 후 이 항목에 추가한다.
- 대안: 공식 기본 patience=100(에폭 100과 같아 조기 종료가 사실상 없음), OOM 지점을 직접 찾아 80%를 곱하는 방식
- 이유: RTX 4090 1장에서 전체 실험 시간을 제한한다. batch=0.80은 "메모리 부족이 나지 않는 최대값의 약 80%"를 공식 옵션으로 구현한 것이다(의미가 완전히 같지는 않다: 배치 최대값이 아니라 메모리 사용률 80%를 목표로 한다).
- 결과가 지지했는가:

## Canny 임계값
- 선택: HR train 타일 밝기 채널의 중앙값 기준으로 한 번 정하고 모든 방법에 같은 값을 쓴다. 확정 값은 Stage 3 이후 이 항목에 추가한다.
- 대안: 방법별·영상별 자동 임계값
- 이유: 방법마다 임계값이 다르면 엣지 지표가 비교 불가능해진다.
- 결과가 지지했는가:

## SR 임시 PNG
- 선택: 공식 SR 스크립트가 GeoTIFF를 읽지 못하면 /root/tmp_sr에 임시 PNG를 만들고 GeoTIFF 변환과 G8 좌표 검사 후 즉시 삭제한다.
- 대안: 공식 스크립트 수정
- 이유: 공식 코드를 고치지 않고, /workspace에 추가 래스터를 남기지 않는다.
- 결과가 지지했는가:

## 환경 분리
- 선택: geo(Python 3.12, 최신 PyTorch), sr(Python 3.10, PyTorch 2.1.2: BasicSR·Real-ESRGAN·SwinIR·ResShift), osediff(Python 3.10, PyTorch 2.0.1)
- 대안: 단일 rsx 환경, 4개 환경(sr과 resshift 분리)
- 이유: ResShift(xformers 0.0.23.post1 → torch==2.1.2)와 OSEDiff(xformers 0.0.20 → torch==2.0.1)가 공식 고정 버전으로 충돌한다. BasicSR 1.4.2는 torchvision<0.17이 필요하고, 최신 rasterio·geopandas는 Python ≥3.11이 필요하다. sr과 ResShift 사이에는 충돌이 없다(docs/env_matrix.md).
- 결과가 지지했는가:

## osediff 환경의 torch 빌드와 추가 고정
- 선택: torch 2.0.1+cu117(PyPI 기본 빌드), torchvision 0.15.2, numpy 1.26.4. 재설치하지 않는다.
- 대안: `--extra-index-url https://download.pytorch.org/whl/cu118`로 torch 2.0.1+cu118 설치
- 이유: OSEDiff requirements.txt는 `torch==2.0.1`만 고정하고 CUDA 빌드는 정하지 않아 PyPI 기본 빌드(cu117)가 설치됐다. RTX 4090(드라이버 CUDA 13.2)에서 `torch.cuda.is_available()` True와 S7 최소 추론을 확인했다(logs/stage0_verify_osediff_20261029_1303.log). torchvision 0.15.2는 torch 2.0.1의 공식 짝으로, basicsr가 버전 없이 끌어오는 torchvision이 torch를 올리지 않게 막고 BasicSR 1.4.2의 torchvision<0.17 조건도 맞춘다. numpy 1.26.4는 NumPy 1.x로 빌드된 torch 2.0.1이 NumPy 2.x에서 깨지지 않게 한다(docs/autofix_log.md 5번).
- 결과가 지지했는가:

## S6 시작 가중치
- 선택: 공식 finetune_realesrgan_x4plus_pairdata.yml을 복사해 pretrain_network_g만 RealESRGAN_x4plus.pth로 바꾸고, netD는 RealESRGAN_x4plus_netD.pth를 쓴다.
- 대안: 공식 yml 그대로(RealESRNet_x4plus에서 시작)
- 이유: S6−S4가 순수한 파인튜닝 효과가 되게 한다. Training.md의 다운로드 안내도 RealESRGAN_x4plus.pth다.
- 결과가 지지했는가:

## SD2.1-base 출처
- 선택: OSEDiff 공식 README가 링크한 미러 Manojb/stable-diffusion-2-1-base. 미러 이름·커밋 해시·SHA256을 third_party/VERSIONS.md에 기록한다.
- 대안: stabilityai/stable-diffusion-2-1-base(2026-10-29 기준 HTTP 401, 접근 불가), S7 제외
- 이유: 규칙상 README 링크에서만 받는다. 원본과 동일성은 검증하지 못하므로 README 한계에 적는다.
- 결과가 지지했는가:

## CPU 병렬 개수
- 선택: GDAL·타일링·평가 프로세스 병렬 상한 12, gdalwarp NUM_THREADS=12, 학습 로더 워커 8(Ultralytics 기본값)
- 대안: 이전 파드 값(병렬 32, NUM_THREADS=32), NUM_THREADS=ALL_CPUS
- 이유: 이 파드의 cgroup CPU 할당량은 13.6코어, RAM 한도는 62GB다. `nproc`은 64를 보고하므로 ALL_CPUS는 과다 구독 위험이 있다. GPU 1장이라 전처리·평가와 학습이 동시에 돌지 않으므로 로더 워커 몫을 따로 뺄 필요가 없고, 시스템 몫 약 1.6코어를 남긴다. 프로세스당 RAM은 약 5GB다.
- 결과가 지지했는가:

## 학습 타일 I/O
- 선택: 학습 직전 해당 학습 타일을 /root/tiles_cache로 복사하고 학습 후 삭제한다.
- 대안: /workspace에서 바로 읽기
- 이유: /workspace는 네트워크 볼륨(FUSE)으로 순차 650 MB/s, 작은 파일 2000개 읽기 4.40s다. /root는 3.4 GB/s, 0.07s다(docs/env_report.md).
- 결과가 지지했는가:
