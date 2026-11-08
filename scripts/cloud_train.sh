#!/usr/bin/env bash
# 클라우드 인스턴스에서 학습을 돌리고, 끝나면 인스턴스를 끈다.
#
#   scripts/cloud_train.sh --out-dir runs/a --shutdown -- --env toy --iterations 200
#   scripts/cloud_train.sh --script scripts/train_combat.py --out-dir runs/combat --shutdown -- --iterations 400
#
# - 항상 --resume 으로 실행한다. 스팟 회수 등으로 꺼졌다 다시 켜지면 마지막 반복 다음부터 이어 간다.
# - 학습이 끝나면 <out-dir>/DONE 을 만든다. 실패하면 <out-dir>/FAILED 를 만든다.
#   둘 중 하나가 있으면 다시 실행해도 학습하지 않는다 (부팅 때마다 다시 도는 것을 막음).
#   다시 돌리려면 그 파일을 지우면 된다.
# - --shutdown 을 줬을 때만 끝난 뒤(성공이든 실패든) 인스턴스를 끈다. 로컬 PC 에서는 빼고 쓴다.
#   SHUTDOWN_DELAY_MIN(기본 1)분 뒤에 꺼지므로, SSH 로 붙어 있다면 `sudo shutdown -c` 로 취소할 수 있다.
#   컨테이너로 빌려 주는 GPU 서비스(vast.ai, RunPod 등)는 안에서 shutdown 해도 과금이 안 멈추므로
#   SHUTDOWN_CMD 에 그 서비스의 정지 명령을 넣는다. 예: SHUTDOWN_CMD='vastai stop instance $CONTAINER_ID'
# - 시그널로 죽은 경우(스팟 회수, 시스템 종료, Ctrl-C)는 실패로 치지 않는다. 다음 실행 때 이어서 학습한다.
set -uo pipefail

usage() {
    echo "사용법: $0 --out-dir DIR [--shutdown] [--python PY] [--script scripts/train.py] -- [학습 인자...]" >&2
    exit 2
}

OUT_DIR=""
SHUTDOWN=0
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=""
SCRIPT="scripts/train.py"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
        --shutdown) SHUTDOWN=1; shift ;;
        --python) PYTHON="${2:-}"; shift 2 ;;
        --script) SCRIPT="${2:-}"; shift 2 ;;
        --) shift; break ;;
        *) usage ;;
    esac
done
[[ -n "$OUT_DIR" ]] || usage
if [[ -z "$PYTHON" ]]; then
    if [[ -x "$REPO/.venv/bin/python" ]]; then PYTHON="$REPO/.venv/bin/python"; else PYTHON="python3"; fi
fi

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/train.log"

power_off() {
    [[ "$SHUTDOWN" == 1 ]] || return 0
    if [[ -n "${SHUTDOWN_CMD:-}" ]]; then
        echo "[cloud_train] 정지 명령 실행: $SHUTDOWN_CMD" | tee -a "$LOG"
        bash -c "$SHUTDOWN_CMD" >>"$LOG" 2>&1
        return 0
    fi
    local delay="${SHUTDOWN_DELAY_MIN:-1}"
    echo "[cloud_train] ${delay}분 뒤 인스턴스를 끕니다 (취소: sudo shutdown -c)" | tee -a "$LOG"
    if [[ $EUID -eq 0 ]]; then shutdown -h "+$delay"; else sudo -n shutdown -h "+$delay"; fi
}

for marker in DONE FAILED; do
    if [[ -e "$OUT_DIR/$marker" ]]; then
        echo "[cloud_train] $OUT_DIR/$marker 가 있어서 학습하지 않습니다. 다시 돌리려면 지우세요." | tee -a "$LOG"
        power_off
        [[ "$marker" == DONE ]] && exit 0 || exit 1
    fi
done

echo "[cloud_train] $(date -Is) 시작: $PYTHON $SCRIPT --resume --out-dir $OUT_DIR $*" | tee -a "$LOG"
# 이 스크립트가 SIGTERM 을 받으면(시스템 종료) 학습 프로세스에도 넘기고, 실패 표시는 하지 않는다.
INTERRUPTED=0
trap 'INTERRUPTED=1; kill -TERM "$CHILD" 2>/dev/null' TERM INT
"$PYTHON" -u "$REPO/$SCRIPT" --resume --out-dir "$OUT_DIR" "$@" >>"$LOG" 2>&1 &
CHILD=$!
wait "$CHILD"
STATUS=$?
# trap 으로 wait 가 먼저 풀렸을 수 있으니 학습 프로세스가 정말 끝날 때까지 기다린다.
while kill -0 "$CHILD" 2>/dev/null; do wait "$CHILD"; STATUS=$?; done

if [[ $STATUS -eq 0 ]]; then
    date -Is >"$OUT_DIR/DONE"
    echo "[cloud_train] $(date -Is) 학습 완료" | tee -a "$LOG"
    power_off
    exit 0
fi
if [[ $INTERRUPTED -eq 1 || $STATUS -gt 128 ]]; then
    echo "[cloud_train] $(date -Is) 중단됨 (종료 코드 $STATUS). 다음 실행 때 이어서 학습합니다." | tee -a "$LOG"
    exit "$STATUS"
fi
echo "exit $STATUS at $(date -Is)" >"$OUT_DIR/FAILED"
echo "[cloud_train] $(date -Is) 학습 실패 (종료 코드 $STATUS). 로그: $LOG" | tee -a "$LOG"
power_off
exit "$STATUS"
