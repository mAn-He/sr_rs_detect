# 정적 검사 결과

2026-10-29 스모크 통과 시점(src/, tests/, run.py, scripts/; third_party 제외), geo 환경.

| 검사 | 명령 | 결과 |
|---|---|---|
| ruff (기본 규칙 E·F, F401·F841 포함) | `ruff check . --exclude third_party` | All checks passed |
| vulture | `vulture src tests run.py scripts --min-confidence 60` | 출력 없음(죽은 코드 없음) |
| pytest | `python -m pytest -q tests` | 7 passed |

Stage 7 마무리 전에 다시 실행해 갱신한다.
