# 환경 보고서

## Stage 0 하드웨어 측정 (2026-10-29)

원본 출력: logs/stage0_hw.log

| 항목 | 측정값 | CLAUDE.md 가정 |
|---|---|---|
| OS | Ubuntu 24.04.5 LTS | Ubuntu |
| GPU | RTX 4090 24564MiB × 1, 드라이버 595.91.07, 드라이버 CUDA 13.2 | RTX 4090 24GB × 1 |
| CPU | `nproc` 256(호스트 전체 표시). cgroup 할당량 cfs_quota 5440000/period 100000 = 54.4 CPU, `RUNPOD_CPU_COUNT=64` | vCPU 8 |
| RAM | `free -h` 251Gi(호스트 표시). cgroup memory.limit 61,999,996,928 B(≈62GB), `RUNPOD_MEM_GB=62` | 62GB |
| 디스크 | `/workspace`는 별도 마운트가 없고 `/`와 같은 overlay 50G(사용 334M)다. `/root`도 같은 파일시스템 | 영구 네트워크 볼륨 `/workspace` |
| /workspace 읽기 | 순차 1MiB O_DIRECT 3.3 GB/s, 4KiB O_DIRECT 130 MB/s(≈32k IOPS). 캐시를 거친 읽기는 11~12 GB/s | - |
| 기존 사용량 | /workspace/.cache/pip 1.1G, /workspace/rs-sr-detect 232K | - |

### 측정값에 따른 판단
- I/O: `/root/tiles_cache`는 `/workspace`와 같은 overlay 디스크라서 복사해도 읽기 속도가 바뀌지 않는다. 순차 3.3 GB/s면 학습 로더의 병목이 아니므로 복사는 생략한다.
- 저장 공간 80% 상한 = 50G × 0.8 = 40G.
