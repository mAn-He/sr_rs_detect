# rs-sr-detect

이 실험은 "PAN이 없는 1.2m급 영상(다른 위성, PAN 미제공 자료)을 0.3m급으로 올려야 하는 상황"의 대리 실험이다.
같은 시점에 같은 위성이(SpaceNet 2, AOI_2_Vegas) 촬영한 실제 멀티스펙트럼 영상(약 1.2m)을 저해상도 입력으로,
팬샤프닝 RGB(0.3m)를 고해상도 정답으로 쓴다. 합성 열화는 쓰지 않는다.

## 한계

- HR(PS-RGB)과 LR(MS) 사이에 남북 약 −0.27 LR px(약 −0.3 m, LR 내용이 북쪽) 오프셋이 지역 전체에 같은 부호로 있다(블록 ECC 중앙값 −0.265, IQR −0.345~−0.199). 원인을 확인하지 못해 보정하지 않았다(docs/decisions.md, docs/data_report.md G5).
- G5 정합 판정(0.5 LR px)에서 test 블록 최대 어긋남이 0.496 LR px(phaseCorrelate, ECC 기준 최대 0.5 미만)로 기준에 0.004 LR px(약 5 mm) 차이로 통과했다. test 24블록 중앙값은 0.252 LR px다. 남북 오프셋 때문에 test 블록 일부는 HR과 LR이 거의 0.5 LR px(0.6 m) 어긋나 있으며, 이 차이는 경로 1과 경로 2 비교에 그대로 남는다(data/full/g5_registration.csv). 0.5를 넘는 train 블록 6개(최대 0.691)는 제외하지 않고 학습에 쓴다.
