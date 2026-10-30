# test 평가 고정 값 (2026-10-30, test 평가 직전 기록)

test 평가는 아래 값으로만 실행한다. 모든 값은 val 블록·train 블록에서 정했고 test 결과를 보고 바꾸지 않는다.
test를 다시 돌리면 아래 '재실행 기록'에 날짜, 이유, 바뀐 값, 커밋을 추가한다.

| 항목 | 값 | 출처 |
|---|---|---|
| 코드 커밋 | 39382f77e447a4f321172c2b6c465dc63e24bea1 | git (SR 단계 결과 커밋, eval 코드 포함) |
| D_HR 가중치 | runs/full/D_HR/weights/best.pt | results/det_D_HR.json (epochs 200 설정, 86에폭 조기 종료, 최적 66에폭, fitness = mAP50-95) |
| D_HR SHA256 | 03b14b45c153797e84a891cf1169e0b5e4107c9893438ca396f0f230fd44c613 | sha256sum |
| D_LR 가중치 | runs/full/D_LR/weights/best.pt | results/det_D_LR.json (epochs 200, 조기 종료 없음) |
| D_LR SHA256 | 78ab26a5d595819144c10b6e699b3f31475f6ee7a33e53cafd36319b8645ec47 | sha256sum |
| S6 가중치 | third_party/Real-ESRGAN/experiments/S6_realesrgan_ft_spacenet/models/net_g_latest.pth (23,600회, 마지막 체크포인트) | results/s6_finetune.json |
| S6 SHA256 | b21fb9c85fd59dc1bccd51563849bc830550ab2f83d36eb749ae023dfbe5923e | sha256sum |
| AP용 conf | 0.001 | configs/default.yaml det_conf_ap |
| max_det | 3000 (val 상한 도달 타일 HR 0 / LR 0) | data/full/label_stats.json, results/det_calibration.json |
| 타일 병합 NMS IoU | 0.5 (val AP50 0.8194 대 IoU 0.7 0.8128, 규칙 "AP50 높은 쪽") | results/det_calibration.json |
| 운영 conf D_HR (HR·모든 SR 행) | 0.5025025025025025 | results/det_calibration.json |
| 운영 conf D_LR (LR 행) | 0.5615615615615616 | results/det_calibration.json |
| 매칭 IoU | 0.5 (UTM 회전 IoU) | configs/default.yaml match_iou |
| 평가 제외 폭 (주) | 8 LR px (9.6 m), 민감도 0 / 8 / 16 LR px | configs/default.yaml eval_buffer_lr, eval_buffer_sens |
| 평가 영역 밖 제외 비율 | 폴리곤 면적의 50% 이상 밖이면 제외 | configs/default.yaml eval_outside_frac |
| 면적 구간 | <50 / 50–150 / 150–400 / >400 m² | configs/default.yaml area_bins |
| 부트스트랩 | test 블록 복원추출 1,000회, 시드 0 | configs/default.yaml bootstrap, seed |
| Canny 임계값 | low 47 / high 94 (HR train 밝기 중앙값 71 × 0.66 / 1.33) | results/canny_thresholds.json |

## 재실행 기록
- 2026-10-30, **사후 분석, 주 결론 아님** (src/posthoc.py, run.py posthoc, 결과 results/posthoc/). 사용자 지시로 test를 다시 사용했다.
  위 고정 값을 하나도 바꾸지 않았다(D_HR SHA256을 스크립트에서 assert, 제외 폭은 주 값 8 LR px만, 부트스트랩은 geo_eval.py와 같은 seed·같은 표본이며
  기존 행 AP50과 CI가 results/geo_eval.csv와 같은지 assert). 새로 탐지한 행은 색 보정 대조군 S0_color, S3_color뿐이다
  (train 블록 MS–PS 쌍에서 학습한 밴드별 선형 변환을 S0·S3 출력에 적용). 내용: S0 대비 이득, 색 보정 대조군, 면적 구간별 재현율·거짓 양성,
  S6 제외 Spearman, 정성 그림. summary.csv와 주 결론은 바꾸지 않는다.
