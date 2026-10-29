"""S2~S7: test LR 타일에 공식 SR 스크립트를 실행하고(공식 환경의 python, subprocess) 결과를 G8 규칙으로 GeoTIFF 저장.
공식 스크립트가 GeoTIFF를 읽지 못하는 S5·S7은 tmp_sr에 임시 PNG 입력을 만들고, 모든 임시 파일은 G8 검사 후 삭제한다.
S0·S1은 gdal_ops.g7(보간)과 tiling(타일 자르기)에서 만든다."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import geopandas as gpd

import gdal_ops as g

TP = g.ROOT / 'third_party'
WT = Path(g.CFG['weights_dir'])
ENABLED = {'S5_resshift_v3_x4': g.CFG['enable_resshift'], 'S7_osediff_x4': g.CFG['enable_osediff']}


def py(env_key):
    return f"{g.CFG['miniforge']}/envs/{g.CFG['envs'][env_key]}/bin/python"


def official(env_key, args, cwd):
    cmd = [py(env_key), *map(str, args)]
    print('$', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def png_inputs(tmp, lr_dir, ids):
    d = tmp / 'in'
    d.mkdir(parents=True)
    for i in ids:
        img = cv2.imread(str(lr_dir / f'{i}.tif'), cv2.IMREAD_COLOR)
        assert img is not None, i
        cv2.imwrite(str(d / f'{i}.png'), img)
    return d


def realesrgan(tmp, lr_dir, name, model_path):
    official('official_sr', [TP / 'Real-ESRGAN/inference_realesrgan.py', '-n', name, '--model_path', model_path,
                             '-i', lr_dir, '-o', tmp / 'out', '--ext', 'png'], TP / 'Real-ESRGAN')
    return lambda i: tmp / 'out' / f'{i}_out.png'


def run_method(m, tmp, lr_dir, hr_dir, ids):
    """공식 스크립트 실행 → 타일 id별 출력 PNG 경로 함수."""
    if m == 'S2_swinir_classical_x4':
        # classical_sr는 GT 폴더를 순회하고 LQ를 '<이름>x4<확장자>'로 찾는다(main_test_swinir.py get_image_pair).
        lq = tmp / 'lq'
        lq.mkdir(parents=True)
        for i in ids:
            (lq / f'{i}x4.tif').symlink_to(lr_dir / f'{i}.tif')
        official('official_sr', [TP / 'SwinIR/main_test_swinir.py', '--task', 'classical_sr', '--scale', 4,
                                 '--training_patch_size', 64, '--model_path',
                                 WT / 'swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth',
                                 '--folder_lq', lq, '--folder_gt', hr_dir], tmp)
        return lambda i: tmp / f'results/swinir_classical_sr_x4/{i}_SwinIR.png'
    if m == 'S3_realesrnet_x4plus':
        return realesrgan(tmp, lr_dir, 'RealESRNet_x4plus', WT / 'realesrgan/RealESRNet_x4plus.pth')
    if m == 'S4_realesrgan_x4plus':
        return realesrgan(tmp, lr_dir, 'RealESRGAN_x4plus', WT / 'realesrgan/RealESRGAN_x4plus.pth')
    if m == 'S6_realesrgan_ft_spacenet':
        net = json.loads((g.W / 's6_finetune.json').read_text())['net_g']
        return realesrgan(tmp, lr_dir, 'RealESRGAN_x4plus', net)
    if m == 'S5_resshift_v3_x4':
        official('official_sr', ['inference_resshift.py', '-i', png_inputs(tmp, lr_dir, ids), '-o', tmp / 'out',
                                 '--task', 'realsr', '--scale', 4, '--version', 'v3'], TP / 'ResShift')
        return lambda i: tmp / 'out' / f'{i}.png'
    if m == 'S7_osediff_x4':
        official('official_osediff', ['test_osediff.py', '-i', png_inputs(tmp, lr_dir, ids), '-o', tmp / 'out',
                                      '--pretrained_model_name_or_path', WT / 'sd21_base',
                                      '--osediff_path', 'preset/models/osediff.pkl',
                                      '--ram_path', WT / 'osediff/ram_swin_large_14m.pth',
                                      '--ram_ft_path', 'preset/models/DAPE.pth'], TP / 'OSEDiff')
        return lambda i: tmp / 'out' / f'{i}.png'
    raise ValueError(m)


def main():
    tiles = gpd.read_file(g.W / 'tiles.geojson')
    ids = sorted(tiles[(tiles.split == 'test') & tiles.use].tile_id)
    lr_dir, hr_dir = g.W / 'tiles/lr/images/test', g.W / 'tiles/hr/images/test'
    fp = gpd.read_file(g.W / 'footprint.geojson').geometry.iloc[0]
    hr_fill, _ = g.fills()
    km2 = len(ids) * (g.CFG['tile_lr'] * g.LR) ** 2 / 1e6
    for m in sys.argv[1:]:
        assert m in g.CFG['sr_methods'], m
        if not ENABLED.get(m, True):
            print(f'{m}: disabled in config, not run')
            continue
        tmp = Path(g.CFG['tmp_sr']) / m
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        t0 = time.time()
        out_png = run_method(m, tmp, lr_dir, hr_dir, ids)
        g.log_sr_time(m, time.time() - t0, km2)
        for i in ids:
            img = cv2.imread(str(out_png(i)), cv2.IMREAD_COLOR)
            assert img is not None, (m, out_png(i))
            g.save_sr(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), lr_dir / f'{i}.tif', hr_dir / f'{i}.tif',
                      g.W / 'sr' / m / 'images/test' / f'{i}.tif', hr_fill, fp)
        shutil.rmtree(tmp)
        print(f'{m}: {len(ids)} tiles saved and checked against the HR grid (G8)')


if __name__ == '__main__':
    main()
