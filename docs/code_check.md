# 정적 검사 결과

## Stage 7 마무리 (2026-10-30, geo 환경, ruff 0.17.3, vulture 2.16)

src/, tests/, run.py, scripts/(third_party 제외). Stage 7에서 바뀐 파일: src/report.py(그림 g·h, readme 모드), src/report_viewer.py, src/posthoc.py, run.py(report 단계).

| 검사 | 명령 | 결과 |
|---|---|---|
| ruff (E·F, F401·F841 포함) | `ruff check . --exclude third_party` | All checks passed |
| vulture | `vulture src tests run.py scripts --min-confidence 60` | 출력 없음(죽은 코드 없음) |
| pytest | `PROJ_DATA=.../share/proj python -m pytest -q tests` | 7 passed |

## 스모크 시점

2026-10-29 스모크 통과 시점(src/, tests/, run.py, scripts/; third_party 제외), geo 환경.

| 검사 | 명령 | 결과 |
|---|---|---|
| ruff (기본 규칙 E·F, F401·F841 포함) | `ruff check . --exclude third_party` | All checks passed |
| vulture | `vulture src tests run.py scripts --min-confidence 60` | 출력 없음(죽은 코드 없음) |
| pytest | `python -m pytest -q tests` | 7 passed |


## README 대조 (Stage 7, `run.py report` → `src/report.py readme`)

README.md·README.ko.md의 결과표를 results/ CSV와 logs/timing.csv에서 다시 쓰고, 두 파일의 수치를 대조한다.
수치 토큰은 코드 블록·인라인 코드·URL·링크 대상·HTML 주석을 뺀 본문과 표에서 뽑는다. 판정 순서:
(1) results/ 아래 CSV·JSON 값과 logs/timing.csv(및 표에 쓴 단계 시간 합)를 README 자릿수로 반올림해 같으면 'results/logs',
(2) 아니면 docs/*.md, configs/default.yaml, third_party/VERSIONS.md 본문에 같은 수치 문자열이 있으면 'docs·config',
(3) 아니면 코드의 허용 목록(인용 연도, arXiv 번호, 라이선스 버전), (4) 나머지는 불일치로 실패 처리한다.
정수(예: 8, 200)는 우연히 일치할 수 있어 (1)의 근거가 약하다. 이 수치들은 사람이 문맥으로 따로 확인했다.
소수도 다른 열의 값과 우연히 맞으면 통과한다. 그래서 본문에 직접 쓴 소수 값(TL;DR, 해석, 사후 분석 문단)은 CSV에서 표와 같은 형식(`f'{v:+.3f}'`)으로 다시 출력해 하나씩 대조했다.
이 수동 대조에서 S3 gain을 본문에 -0.016 [-0.035, +0.004]로 적은 오류를 찾아 표 값 -0.015 [-0.035, +0.003]으로 고쳤다. 자동 검사는 이 오류를 잡지 못했다.

<!-- BEGIN:readme_check -->
- README.md: 수치 토큰 962개 — results/logs 일치 935, docs·config 일치 24, 허용 목록 3, 불일치 0
  - results/logs 밖 수치: -0.125 (docs/config), -0.199 (docs/config), -0.345 (docs/config), 1,168 (docs/config)×3, 1,200 (docs/config), 13.6 (docs/config), 153.6 (docs/config), 162.5 (docs/config)×2, 1812.04098 (allow), 2019 (allow), 23,394 (docs/config), 24,564 (docs/config), 2605.00310 (allow), 3,850 (docs/config), 3.10 (docs/config)×2, 32611 (docs/config), 4090 (docs/config)×2, 5,020 (docs/config), 5,194 (docs/config), 576 (docs/config), 7,120 (docs/config), −2 (docs/config)
- README.ko.md: 수치 토큰 962개 — results/logs 일치 935, docs·config 일치 24, 허용 목록 3, 불일치 0
- 두 파일 수치 토큰 다중집합 일치: True
- 자동 생성 표 11개(main, quality, recall, buffer, spearman, ph_s0, ph_colorfit, ph_color, ph_area, ph_spearman, timing): 두 파일 동일, CSV에서 생성
- 그림 경로 순서까지 일치: True (7개)
- 제목 구조(#, ##, ###) 일치: True (11개)
- README.md 상대 링크 33개, 없는 대상 0
- README.ko.md 상대 링크 33개, 없는 대상 0
<!-- END:readme_check -->
