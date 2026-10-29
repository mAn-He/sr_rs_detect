# 환경 요구사항 매트릭스 (E1)

조사일: 2026-10-29. 모든 값은 아래 출처 URL의 공식 파일에서 직접 확인했다. "명시 없음"은 공식 문서에 값이 없다는 뜻이다.

하드웨어 사전 확인: RTX 4090(sm_89), NVIDIA 드라이버 595.91.07, 드라이버 CUDA 13.2 → cu118/cu12x/cu13x PyTorch 휠 모두 실행 가능.

## 구성 요소별 공식 요구사항

| 구성 요소 | Python | PyTorch | torchvision | CUDA | 고정 버전 요구 | 추가 의존성 | 출처 URL |
|---|---|---|---|---|---|---|---|
| GDAL / libgdal (conda-forge) | 명시 없음 (conda-forge 빌드) | - | - | - | 최신 3.13.3 | conda-forge 설치 | https://anaconda.org/conda-forge/gdal |
| rasterio | ≥3.12 (1.5.2) | - | - | - | numpy≥2 (1.5.2) | GDAL | https://pypi.org/project/rasterio/ |
| geopandas | ≥3.11 (1.2.0) | - | - | - | 없음 | shapely, pyproj | https://pypi.org/project/geopandas/ |
| shapely | ≥3.11 (2.2.0) | - | - | - | 없음 | GEOS | https://pypi.org/project/shapely/ |
| OpenCV (opencv-python) | ≥3.6 | - | - | - | Ultralytics: ≥4.7.0, ≠4.13.0.90 | - | https://pypi.org/project/opencv-python/ |
| Ultralytics 8.4.192 (YOLO26-OBB) | ≥3.8 | ≥1.8.0 | ≥0.9.0 | 명시 없음 | numpy≥1.23.0 | 가중치 `yolo26s-obb.pt`(assets v8.4.0, DOTAv1 사전학습) | https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml , https://docs.ultralytics.com/tasks/obb/ |
| BasicSR 1.4.2 (PyPI 최신) | ≥3.7 | ≥1.7 | 명시 없음, **실제로 <0.17 필요** | 명시 없음 | 1.4.2의 `basicsr/data/degradations.py:8`이 `torchvision.transforms.functional_tensor`를 import(torchvision 0.17에서 제거됨). master 브랜치는 수정됐지만 PyPI 미배포 | addict, lmdb, scikit-image, tb-nightly 등 | https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md |
| Real-ESRGAN 0.3.0 | 명시 없음 | ≥1.7 | 명시 없음 | 명시 없음 | basicsr≥1.4.2 | facexlib≥0.2.5, gfpgan≥1.3.5 | https://github.com/xinntao/Real-ESRGAN/blob/master/requirements.txt |
| SwinIR | 명시 없음 | 명시 없음 | - | 명시 없음 | 없음(requirements 파일 없음) | timm(`models/network_swinir.py` import) | https://github.com/JingyunLiang/SwinIR |
| ResShift (기본 브랜치 `journal`) | 3.10 (env.yml 3.10.13) | 2.1.2+cu118 (env.yml) / requirements.txt는 2.1.1 | 0.16.0 (env.yml) | 11.8 | xformers 0.0.23(.post1, torch==2.1.2 고정) | timm, lpips, omegaconf, loralib, albumentations, basicsr(`load_file_from_url` import) | https://github.com/zsyOAOA/ResShift/tree/journal |
| OSEDiff | 3.10 | ==2.0.1 | 명시 없음(2.0.1 짝은 0.15.2) | 명시 없음 | xformers==0.0.20(torch==2.0.1 고정), diffusers==0.25.0, transformers==4.28.1, peft==0.9.0, huggingface_hub==0.25.0, open-clip-torch==2.20.0, Pillow==9.5.0, pyiqa==0.1.10 | basicsr, loralib, fairscale; SD2.1-base, RAM, DAPE | https://github.com/cswry/OSEDiff |
| AdcSR (비활성) | 3.10 | ==2.4.1 | ==0.19.1 | 명시 없음 | xformers==0.0.28.post1, diffusers==0.32.2, transformers==4.37.2, peft==0.13.2, pyiqa==0.1.13 | pytorch-lightning 2.4.0 | https://github.com/Guaishou74851/AdcSR |

## 공식 가중치와 의존 모델 링크 확인 (HTTP 상태, 2026-10-29)

| ID | 파일 | 출처 | 상태 |
|---|---|---|---|
| D_HR/D_LR | yolo26s-obb.pt | github.com/ultralytics/assets/releases/download/v8.4.0/ | 200 |
| S2 | 001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth | github.com/JingyunLiang/SwinIR/releases/download/v0.0/ | 200 |
| S3 | RealESRNet_x4plus.pth | Real-ESRGAN releases v0.1.1 | 200 |
| S4 | RealESRGAN_x4plus.pth | Real-ESRGAN releases v0.1.0 | 200 |
| S6 | RealESRGAN_x4plus_netD.pth | Real-ESRGAN releases v0.2.2.3 | 200 |
| S5 | resshift_realsrx4_s4_v3.pth, autoencoder_vq_f4.pth | ResShift releases v2.0 | 200 |
| S7 | osediff.pkl, DAPE.pth | OSEDiff 저장소 `preset/models/` (git 포함) | 존재 |
| S7 | ram_swin_large_14m.pth | huggingface.co/spaces/xinyu1205/recognize-anything (README 링크) | 200 |
| S7 | SD2.1-base | README 링크: huggingface.co/Manojb/stable-diffusion-2-1-base | 200 (개인 미러, 2025-06-13 생성, openrail++) |
| 참고 | stabilityai/stable-diffusion-2-1-base | HF API | 401 (원 저장소 접근 불가) |

## 공식 사용법에서 확인한 실행 세부

- ResShift: `inference_resshift.py --task realsr --scale 4 --version v3` (`journal` 브랜치 README와 일치). 가중치를 cwd 기준 `./weights`에 받는다. `--chop_size`, `--chop_stride`, `--bs` 옵션이 있다.
- OSEDiff: `test_osediff.py`는 폴더 입력에서 `*.png`만 읽는다(PIL). 기본값은 `--process_size 512`, `--align_method adain`, `--mixed_precision fp16`이고 latent/VAE 타일 옵션이 있다.
- SwinIR: `--task classical_sr`이면 `--folder_gt`를 순회하면서 `{folder_lq}/{name}x4{ext}`로 LQ를 찾는다. GT 폴더가 반드시 필요하다. `--tile` 옵션이 있다.
- Real-ESRGAN: `inference_realesrgan.py -n RealESRNet_x4plus|RealESRGAN_x4plus --model_path ... -t <tile>`. 기본 fp16이다. cv2로 읽으므로 8비트 GeoTIFF를 직접 읽을 수 있다.
- Real-ESRGAN paired 파인튜닝: `options/finetune_realesrgan_x4plus_pairdata.yml`. total_iter 400000, batch_size_per_gpu 12, gt_size 256, `RealESRGANPairedDataset`(disk). 단일 GPU 명령은 `python realesrgan/train.py -opt ... --auto_resume`.
- Ultralytics 기본값: epochs 100, patience 100, batch 16(`batch` 0~1 float이면 GPU 메모리 비율 AutoBatch), workers 8, cache False, amp True, seed 0, deterministic True.

## 충돌 분석 (E2)

1. **ResShift ↔ OSEDiff: 확정 충돌.** xformers 0.0.23.post1은 torch==2.1.2를, xformers 0.0.20은 torch==2.0.1을 정확히 요구한다. 두 모델 모두 공식 고정 버전이라 한 환경에 둘 수 없다.
2. **BasicSR 1.4.2 ↔ 최신 PyTorch: 확정 충돌.** torchvision ≥0.17(torch ≥2.2)에서 BasicSR 1.4.2를 import하면 실패한다. ResShift(torch 2.1.2/torchvision 0.16)와 OSEDiff(torch 2.0.1/torchvision 0.15)는 이 범위 안이다.
3. **GDAL 계열 최신 버전 ↔ Python 3.10.** rasterio 1.5.2는 Python ≥3.12, geopandas 1.2·shapely 2.2는 ≥3.11을 요구한다. Python 3.10에 두려면 구버전을 써야 한다.
4. **OSEDiff ↔ AdcSR(비활성):** torch 2.0.1과 2.4.1로 서로 충돌한다. 활성화하면 별도 환경이 필요하다.

결론: 단일 `rsx` 환경은 공식 고정 버전 기준으로 불가능하다(1번만으로 확정). E4 분리안으로 간다.

## 분리안 (E3 검증으로 확정)

| 환경 | Python | PyTorch | 담당 | 근거 |
|---|---|---|---|---|
| geo | 3.12 | 최신 안정판 + cu12x | GDAL·rasterio·geopandas·shapely·OpenCV, Ultralytics(D_HR/D_LR 학습·추론), S0/S1, 병합·평가·LPIPS·그림, run.py | 최신 GDAL 계열과 Ultralytics를 같은 환경에서 쓴다 |
| sr | 3.10 | 2.1.2+cu118, torchvision 0.16.2, xformers 0.0.23.post1 | BasicSR 1.4.2, Real-ESRGAN(S3, S4, S6 학습·추론), SwinIR(S2), ResShift(S5) | ResShift 공식 고정 버전이 BasicSR 1.4.2 허용 범위(torchvision<0.17) 안에 있다. Real-ESRGAN·SwinIR은 고정 버전이 없다 |
| osediff | 3.10 | 2.0.1+cu118 | OSEDiff(S7) | requirements.txt를 그대로 설치한다 |

CLAUDE.md의 예상 후보(geo, sr, resshift, osediff 4개)와 다른 점은 sr과 resshift를 합친 것이다. 둘 사이에 충돌하는 고정 버전이 없기 때문이다. verify_env_sr.py에서 S2~S5 최소 추론이 하나라도 실패하면 resshift를 따로 분리한다.
