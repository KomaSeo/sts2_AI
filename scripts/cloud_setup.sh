#!/usr/bin/env bash
# 새 클라우드 서버(AWS 등)에서 한 번 실행: 설치 → zhiyue 시뮬레이터 받기 → 테스트 → GPU 시험 학습.
#
#   bash scripts/cloud_setup.sh                 # 이 저장소 폴더 안에서
#   bash scripts/cloud_setup.sh --bench         # 기본 설정으로 실제 반복 1회 시간도 잰다 (몇 분)
#
# - Python 3.12 는 uv 로 설치한다 (Ubuntu 22.04 의 3.10 은 zhiyue 가 요구하는 3.11 보다 낮다).
# - GPU(nvidia-smi)가 있으면 CUDA 용 torch, 없으면 CPU 용 torch 를 설치한다.
# - zhiyue/sts2-rl-agent 는 이 저장소 옆(../sts2-rl-agent)에 받고, 검증한 커밋으로 고정한다.
#   그 저장소의 코드는 이 저장소에 넣지 않는다 (README "실제 StS2 전투로 학습" 참고).
# - 여러 번 실행해도 된다 (이미 있는 것은 건너뜀).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZHIYUE_DIR="$(cd "$REPO/.." && pwd)/sts2-rl-agent"
ZHIYUE_URL="https://github.com/zhiyue/sts2-rl-agent.git"
ZHIYUE_COMMIT="5d874e8"  # 이 저장소에서 어댑터 테스트를 통과한 커밋
BENCH=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bench) BENCH=1; shift ;;
        --zhiyue-dir) ZHIYUE_DIR="$2"; shift 2 ;;
        --zhiyue-commit) ZHIYUE_COMMIT="$2"; shift 2 ;;
        *) echo "알 수 없는 인자: $1" >&2; exit 2 ;;
    esac
done
step() { echo; echo "=== $* ==="; }
cd "$REPO"

step "1/6 uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

step "2/6 Python 3.12 가상환경 (.venv)"
[[ -x .venv/bin/python ]] || uv venv --python 3.12 .venv
PY="$REPO/.venv/bin/python"
"$PY" --version

step "3/6 패키지"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    GPU=1
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    uv pip install --python "$PY" torch  # PyPI 의 리눅스 torch 는 CUDA 포함
else
    GPU=0
    echo "GPU 가 없습니다 (nvidia-smi 실패). CPU 용 torch 를 설치합니다."
    uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cpu
fi
uv pip install --python "$PY" -r requirements.txt gymnasium
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if [[ $GPU == 1 ]] && ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "GPU 는 있는데 torch 가 CUDA 를 못 씁니다. NVIDIA 드라이버가 깔린 AMI(Deep Learning AMI 등)인지 확인하세요." >&2
    exit 1
fi

step "4/6 zhiyue/sts2-rl-agent ($ZHIYUE_COMMIT)"
if [[ ! -d "$ZHIYUE_DIR/.git" ]]; then
    git clone -q "$ZHIYUE_URL" "$ZHIYUE_DIR"
fi
git -C "$ZHIYUE_DIR" fetch -q origin || true
git -C "$ZHIYUE_DIR" checkout -q "$ZHIYUE_COMMIT"
git -C "$ZHIYUE_DIR" log -1 --format='%h %ci %s'

step "5/6 테스트 (어댑터 포함)"
ZHIYUE_PATH="$ZHIYUE_DIR" "$PY" -m pytest -q -p no:cacheprovider tests 2>&1 | tail -3

step "6/6 시험 학습 (작은 설정 1회)"
DEV=$([[ $GPU == 1 ]] && echo cuda || echo cpu)
SMOKE="$(mktemp -d)"
"$PY" scripts/train_combat.py --env zhiyue --zhiyue-path "$ZHIYUE_DIR" --device "$DEV" \
    --iterations 1 --group-size 2 --focus-per-iter 4 --deck-max 15 --relics-max 10 --eval-every 1 --eval-rounds 1 \
    --out-dir "$SMOKE" 2>&1 | grep -v "^DEBUG" | cut -c1-200
rm -rf "$SMOKE"

if [[ $BENCH == 1 ]]; then
    step "벤치마크: 기본 설정(유물 0~40)으로 반복 1회"
    BDIR="$(mktemp -d)"
    "$PY" scripts/train_combat.py --env zhiyue --zhiyue-path "$ZHIYUE_DIR" --device "$DEV" \
        --iterations 1 --relics-max 40 --eval-every 0 --out-dir "$BDIR" 2>&1 | grep -E "^iter=" | grep -oE "sec=[0-9.]+"
    rm -rf "$BDIR"
fi

echo
echo "준비 완료. device=$DEV, zhiyue=$ZHIYUE_DIR"
echo "다음: scripts/cloud_sweep.sh 로 실험을 돌리세요 (docs/aws.md)."
