"""sr 환경 검증: import, GPU, S2(SwinIR)·S3/S4(Real-ESRGAN)·S5(ResShift) 공식 스크립트 최소 추론(샘플 1장)."""
import shutil
import subprocess
import sys
from pathlib import Path

import basicsr
import cv2
import realesrgan.version
import torch
import torchvision
import xformers
import yaml

root = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((root / 'configs/default.yaml').read_text())
w = Path(cfg['weights_dir'])
tp = root / 'third_party'
out = Path(cfg['tmp_sr']) / 'verify_sr'
for m in (torch, torchvision, xformers, basicsr, realesrgan.version, cv2):
    print(f'{m.__name__:12s} {m.__version__}')
print('cuda_available', torch.cuda.is_available(), 'torch cuda', torch.version.cuda, torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
assert 'RTX 4090' in torch.cuda.get_device_name(0)
shutil.rmtree(out, ignore_errors=True)


def run(cmd, cwd):
    print('$', ' '.join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def check(lq_path, sr_path):
    lq, sr = cv2.imread(str(lq_path)), cv2.imread(str(sr_path))
    assert sr is not None, sr_path
    assert sr.shape[:2] == (lq.shape[0] * 4, lq.shape[1] * 4), (lq.shape, sr.shape)
    print('OK', sr_path, lq.shape, '->', sr.shape)


# S2 SwinIR: classical_sr는 GT 폴더를 순회하므로 샘플 1장으로 GT/LQ 폴더를 만든다(README 명령과 같은 인자).
gt, lq = out / 'swinir_gt', out / 'swinir_lq'
gt.mkdir(parents=True)
lq.mkdir()
shutil.copy(tp / 'SwinIR/testsets/Set5/HR/baby.png', gt)
shutil.copy(tp / 'SwinIR/testsets/Set5/LR_bicubic/X4/babyx4.png', lq)
run([sys.executable, tp / 'SwinIR/main_test_swinir.py', '--task', 'classical_sr', '--scale', 4,
     '--training_patch_size', 64, '--model_path', w / 'swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth',
     '--folder_lq', lq, '--folder_gt', gt], cwd=out)
check(lq / 'babyx4.png', out / 'results/swinir_classical_sr_x4/baby_SwinIR.png')

# S3, S4 Real-ESRGAN
sample = tp / 'Real-ESRGAN/inputs/0014.jpg'
for name in ('RealESRNet_x4plus', 'RealESRGAN_x4plus'):
    run([sys.executable, tp / 'Real-ESRGAN/inference_realesrgan.py', '-n', name,
         '--model_path', w / f'realesrgan/{name}.pth', '-i', sample, '-o', out / name], cwd=tp / 'Real-ESRGAN')
    check(sample, out / name / '0014_out.jpg')

# S5 ResShift (cwd 기준 ./weights)
if cfg['enable_resshift']:
    sample = tp / 'ResShift/testdata/RealSet65/00003.png'
    run([sys.executable, 'inference_resshift.py', '-i', sample, '-o', out / 'resshift',
         '--task', 'realsr', '--scale', 4, '--version', 'v3'], cwd=tp / 'ResShift')
    check(sample, out / 'resshift' / f'{sample.stem}.png')

shutil.rmtree(out)
print('VERIFY sr OK')
