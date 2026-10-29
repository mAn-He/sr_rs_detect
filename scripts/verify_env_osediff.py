"""osediff 환경 검증: import, GPU, S7 OSEDiff 공식 test_osediff.py 최소 추론(샘플 1장)."""
import shutil
import subprocess
import sys
from pathlib import Path

import diffusers
import torch
import transformers
import xformers
import yaml
from PIL import Image

root = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((root / 'configs/default.yaml').read_text())
w = Path(cfg['weights_dir'])
repo = root / 'third_party/OSEDiff'
out = Path(cfg['tmp_sr']) / 'verify_osediff'
for m in (torch, xformers, diffusers, transformers):
    print(f'{m.__name__:12s} {m.__version__}')
print('cuda_available', torch.cuda.is_available(), 'torch cuda', torch.version.cuda, torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
assert 'RTX 4090' in torch.cuda.get_device_name(0)
shutil.rmtree(out, ignore_errors=True)

sample = repo / 'preset/datasets/test_dataset/input/0801_pch_00001.png'
cmd = [sys.executable, 'test_osediff.py', '-i', sample, '-o', out,
       '--pretrained_model_name_or_path', w / 'sd21_base', '--osediff_path', 'preset/models/osediff.pkl',
       '--ram_path', w / 'osediff/ram_swin_large_14m.pth', '--ram_ft_path', 'preset/models/DAPE.pth']
print('$', ' '.join(map(str, cmd)), flush=True)
subprocess.run([str(c) for c in cmd], cwd=repo, check=True)
lq, sr = Image.open(sample), Image.open(out / sample.name)
assert sr.size == (lq.size[0] * 4, lq.size[1] * 4), (lq.size, sr.size)
print('OK', lq.size, '->', sr.size)
shutil.rmtree(out)
print('VERIFY osediff OK')
