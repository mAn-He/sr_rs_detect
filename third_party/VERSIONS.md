# 외부 저장소 버전

| 저장소 | URL | 커밋 해시 | 클론 일시 |
|---|---|---|---|
| SwinIR | https://github.com/JingyunLiang/SwinIR | 6545850fbf8df298df73d81f3e8cba638787c8bd | 2026-10-29 10:47 UTC |
| Real-ESRGAN | https://github.com/xinntao/Real-ESRGAN | a4abfb2979a7bbff3f69f58f58ae324608821e27 | 2026-10-29 10:47 UTC |
| BasicSR | https://github.com/XPixelGroup/BasicSR | 8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a | 2026-10-29 10:48 UTC |
| ResShift | https://github.com/zsyOAOA/ResShift | bb03b7d21614cace01787e097c8a6ab6b945227d | 2026-10-29 10:51 UTC |
| OSEDiff | https://github.com/cswry/OSEDiff | cdd3b7b04850fa9bb7faf8f2187edd279cc96507 | 2026-10-29 10:53 UTC |

# Hugging Face 의존 모델 (OSEDiff)

SD2.1-base는 OSEDiff README가 링크한 개인 미러에서 받았다. stabilityai 원본과 같은 파일인지는 검증하지 않았다.

| 모델 | HF 저장소 | 커밋 해시 | 받은 시각 |
|---|---|---|---|
| SD2.1-base | Manojb/stable-diffusion-2-1-base | 0094d483a120f3f33dafbd187ea4aa60d10de75c | 2026-10-29 12:51 UTC |
| bert-base-uncased (RAM 토크나이저) | google-bert/bert-base-uncased | 86b5e0934494bd15c9632b12f734a8a67f723594 | 2026-10-29 12:51 UTC |

| 파일 | SHA256 |
|---|---|
| sd21_base/text_encoder/model.safetensors | cce6febb0b6d876ee5eb24af35e27e764eb4f9b1d0b7c026c8c3333d4cfc916c |
| sd21_base/unet/diffusion_pytorch_model.safetensors | 6dfae3e5f7d459b50f4b0850ead945972c75bb0e1897628933e169eb43974214 |
| sd21_base/vae/diffusion_pytorch_model.safetensors | a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815 |
| osediff/ram_swin_large_14m.pth (xinyu1205/recognize-anything Space) | 15c729c793af28b9d107c69f85836a1356d76ea830d4714699fb62e55fcc08ed |
| OSEDiff preset/models/osediff.pkl (저장소 포함) | 94821389ce33e2bd314e6e782547b26fcff248715993ad9c754c06f7e6c27804 |
| OSEDiff preset/models/DAPE.pth (저장소 포함) | a7028be2edcbe9ab0bd1c4ab6f2a2a86f4b44d32261a4faa50ae10fdd9b2feba |
