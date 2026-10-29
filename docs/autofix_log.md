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
