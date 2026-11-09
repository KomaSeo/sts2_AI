#!/usr/bin/env bash
# 실험 여러 개를 한 서버에서 동시에 돌리고, 전부 끝나면 (선택) 서버를 끈다.
#
#   scripts/cloud_sweep.sh --plan experiments/size_lr.txt --runs-dir runs/sweep --shutdown
#
# 계획 파일 형식 (# 뒤는 주석):
#   @script scripts/train_combat.py        모든 실험이 쓸 학습 스크립트 (기본 scripts/train_combat.py)
#   @common --env zhiyue --relics-max 40   모든 실험에 붙일 인자 (여러 줄이면 이어 붙임)
#   이름  인자...                           실험 하나. 결과는 <runs-dir>/<이름>/
#
# - 실험마다 scripts/cloud_train.sh 로 돌린다: 항상 --resume, 끝나면 DONE, 실패하면 FAILED 표시.
#   그래서 스팟 회수 뒤 다시 실행하면 끝난 실험은 건너뛰고 나머지만 이어서 돈다.
# - --parallel N: 동시에 돌릴 실험 수 (기본: 전부). 실험 하나가 CPU 1코어를 쓴다.
# - --shutdown: 모든 실험이 DONE/FAILED 가 되면 끈다 (SHUTDOWN_CMD, SHUTDOWN_DELAY_MIN 은 cloud_train.sh 와 같음).
#   시그널로 중단된 경우(스팟 회수, 시스템 종료)에는 끄지 않는다.
#   이번 실행에서 돌린 실험이 하나도 없으면(이미 모두 끝남) 끄지 않는다: 결과를 받으려고 서버를 다시 켰을 때
#   부팅하자마자 다시 꺼지지 않게.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAN="" RUNS="" PARALLEL=0 SHUTDOWN=0 PYTHON=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --plan) PLAN="$2"; shift 2 ;;
        --runs-dir) RUNS="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --shutdown) SHUTDOWN=1; shift ;;
        --python) PYTHON="$2"; shift 2 ;;
        *) echo "사용법: $0 --plan FILE --runs-dir DIR [--parallel N] [--shutdown] [--python PY]" >&2; exit 2 ;;
    esac
done
[[ -f "$PLAN" && -n "$RUNS" ]] || { echo "--plan 파일과 --runs-dir 가 필요합니다" >&2; exit 2; }
mkdir -p "$RUNS"
LOG="$RUNS/sweep.log"
log() { echo "[cloud_sweep] $(date -Is) $*" | tee -a "$LOG"; }

SCRIPT="scripts/train_combat.py"
COMMON=()
NAMES=()
declare -A ARGS
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    read -r -a words <<<"$line"
    [[ ${#words[@]} -eq 0 ]] && continue
    case "${words[0]}" in
        @script) SCRIPT="${words[1]}" ;;
        @common) COMMON+=("${words[@]:1}") ;;
        *) NAMES+=("${words[0]}"); ARGS["${words[0]}"]="${words[*]:1}" ;;
    esac
done <"$PLAN"
[[ ${#NAMES[@]} -gt 0 ]] || { echo "계획 파일에 실험이 없습니다" >&2; exit 2; }
[[ $PARALLEL -gt 0 ]] || PARALLEL=${#NAMES[@]}
PY_ARGS=()
[[ -n "$PYTHON" ]] && PY_ARGS=(--python "$PYTHON")

power_off() {
    [[ "$SHUTDOWN" == 1 ]] || return 0
    if [[ -n "${SHUTDOWN_CMD:-}" ]]; then
        log "정지 명령 실행: $SHUTDOWN_CMD"
        bash -c "$SHUTDOWN_CMD" >>"$LOG" 2>&1
        return 0
    fi
    local delay="${SHUTDOWN_DELAY_MIN:-1}"
    log "${delay}분 뒤 서버를 끕니다 (취소: sudo shutdown -c)"
    if [[ $EUID -eq 0 ]]; then shutdown -h "+$delay"; else sudo -n shutdown -h "+$delay"; fi
}

INTERRUPTED=0
trap 'INTERRUPTED=1; kill -TERM $(jobs -p) 2>/dev/null' TERM INT
log "실험 ${#NAMES[@]}개 (동시 $PARALLEL): ${NAMES[*]}"
LAUNCHED=0
for name in "${NAMES[@]}"; do
    if [[ -e "$RUNS/$name/DONE" || -e "$RUNS/$name/FAILED" ]]; then
        log "건너뜀: $name (이미 $(ls "$RUNS/$name" | grep -E '^(DONE|FAILED)$'))"
        continue
    fi
    while [[ $(jobs -rp | wc -l) -ge $PARALLEL ]]; do wait -n; done
    [[ $INTERRUPTED == 1 ]] && break
    LAUNCHED=$((LAUNCHED + 1))
    # shellcheck disable=SC2206
    extra=(${ARGS[$name]})
    "$REPO/scripts/cloud_train.sh" --script "$SCRIPT" --out-dir "$RUNS/$name" "${PY_ARGS[@]}" \
        -- "${COMMON[@]}" "${extra[@]}" >/dev/null 2>&1 &
    log "시작: $name (로그 $RUNS/$name/train.log)"
done
while [[ -n "$(jobs -rp)" ]]; do wait -n; done

done_n=0 failed=() pending=()
for name in "${NAMES[@]}"; do
    if [[ -e "$RUNS/$name/DONE" ]]; then done_n=$((done_n + 1))
    elif [[ -e "$RUNS/$name/FAILED" ]]; then failed+=("$name")
    else pending+=("$name"); fi
done
log "끝: 완료 $done_n, 실패 ${#failed[@]} ${failed[*]:-}, 미완료 ${#pending[@]} ${pending[*]:-}"
if [[ $INTERRUPTED == 1 || ${#pending[@]} -gt 0 ]]; then
    log "중단된 실험이 있어 서버를 끄지 않습니다. 다시 실행하면 이어서 돕니다."
    exit 1
fi
if [[ $LAUNCHED == 0 ]]; then
    log "이번에 돌린 실험이 없어(이미 모두 끝남) 서버를 끄지 않습니다."
else
    power_off
fi
[[ ${#failed[@]} -eq 0 ]]
