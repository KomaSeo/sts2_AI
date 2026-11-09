# sts2_AI

Slay the Spire 2를 플레이하는 AI의 구조 실험입니다.

- 게임의 각 요소(덱, 유물·포션, 상태, 맵, 전투)를 **각각 인코딩하는 인코더**
- 공개된 맵의 **각 노드와 길에서 죽을 확률**을 예측하는 평가 에이전트
- 아직 맵이 공개되지 않은 **다음 막들에서 죽을 확률**을 예측하는 평가 에이전트
- 같은 전투를 여러 번 굴리고 평가 에이전트의 점수로 서로 비교해 학습하는 **전투 에이전트 (GRPO)**

시뮬레이터로는 [r33hab/sts2](https://github.com/r33hab/sts2)(MIT, C# NativeAOT + Python)를 쓰도록 어댑터를 만들었습니다. 다만 이 저장소를 만든 환경에서는 그 시뮬레이터를 빌드할 수 없어서, 파이프라인 전체는 같은 모양의 **토이 덱빌딩 게임**(`sts2ai/envs/toy.py`)으로 검증했습니다.

## 구조

```
                ┌─────────── SharedEmbeddings ───────────┐
                │ card / relic / potion / buff / enemy /  │
                │ intent / encounter / node_type / act    │
                └──────────────────┬──────────────────────┘
     ┌──────────┬──────────┬───────┴──────┬────────────────┐
 DeckEncoder ItemEncoder StatusEncoder  MapEncoder      CombatEncoder
 (카드 집합)  (유물+포션)  (HP·골드·층)   (DAG, 양방향     (전투 토큰
                                         메시지 전달)      Transformer)
     └──────────┴──────────┘               │                 │
        RunEncoder (런 맥락 ctx) ──FiLM──▶ 노드 벡터          │
               │                           │                 │
     ┌─────────┴──────────┐    ┌───────────┘                 │
 FutureActHead        NodeRiskHead                      CombatPolicy
 P(막 k에서 사망)       hazard[v] = P(노드 v에서 사망)        행동 포인터 + 타깃 포인터
 (k > 현재 막)          survive[v] = 막을 살아서 끝낼 확률
     └─────────┬──────────┘
        log V = log P(런 클리어)  ──── GRPO 점수 ────▶ CombatPolicy 학습
```

| 파일 | 내용 |
|---|---|
| `sts2ai/schema.py` | 시뮬레이터와 무관한 공통 데이터 구조 (`RunState`, `MapGraph`, `CombatObs`) |
| `sts2ai/encoders/embeddings.py` | 공유 임베딩 테이블, 집합 Transformer |
| `sts2ai/encoders/deck.py` | 덱 인코더: 순서 없는 카드 집합, 요약 벡터 + 카드별 토큰 |
| `sts2ai/encoders/items.py` | 유물(카운터 포함) + 포션 인코더 |
| `sts2ai/encoders/status.py` | HP, 골드, 층, 승천, 막 |
| `sts2ai/encoders/map.py` | 맵 GNN: 후속 노드 방향(앞으로 올 위험) + 선행 노드 방향(누적 소모), 런 맥락을 FiLM으로 주입 |
| `sts2ai/encoders/combat.py` | 전투 토큰 Transformer (위치 인코딩 없음, 뽑을 더미 순서를 표현할 수 없음) |
| `sts2ai/encoders/run.py` | 위 인코더들을 합친 런 트렁크 |
| `sts2ai/agents/evaluator.py` | 노드/길 위험도 평가기 + 미래 막 위험도 평가기 |
| `sts2ai/agents/combat_policy.py` | 전투 에이전트 |
| `sts2ai/train/grpo.py` | GRPO: 분기 롤아웃, 채점, 그룹 상대 이점, clipped 업데이트 |
| `sts2ai/train/evaluator_train.py` | 생존 분석식 라벨 만들기와 평가기 학습 |
| `sts2ai/train/loop.py` | 전체 학습 루프 |
| `sts2ai/train/combat_train.py` | 전투만 따로 학습 (무작위 덱·HP·포션, 같은 시드 GRPO) |
| `sts2ai/envs/toy.py` | 검증용 토이 게임 |
| `sts2ai/envs/sts2_adapter.py` | r33hab/sts2 어댑터 (**실행 검증 전**) |
| `sts2ai/envs/zhiyue_adapter.py` | zhiyue/sts2-rl-agent 어댑터: 실제 StS2 전투로 전투 에이전트만 따로 학습 |

## 평가 에이전트

**노드/길 평가기 (현재 막)**
- `hazard[v]`: 지금 상태로 출발해 노드 v에 도착했을 때 그 방에서 죽을 확률
- `survive[v] = (1 - hazard[v]) · max_{u ∈ 다음(v)} survive[u]`: v로 간 뒤 남은 길을 가장 안전하게 고르면 이 막을 살아서 끝낼 확률
- 길 하나의 사망 확률: `1 - Π(1 - hazard)` (`Evaluator.path_death`), 가장 안전한 길: `Evaluator.best_path`
- 맵에서 다음 노드는 `survive`가 가장 높은 곳으로 고릅니다 (학습 중에는 ε 확률로 무작위).

**미래 막 평가기**
- `p_act[k]`: 막 k에 도착한다면 그 막에서 죽을 확률. 현재 막보다 뒤의 막에만 값이 있습니다.
- **마지막 막(3막)에서는 남은 막이 없어서 출력이 비고, 미래 사망 확률은 0입니다.** 3막의 위험은 맵이 공개돼 있으니 노드 평가기가 맡습니다.

**라벨 (생존 분석 방식)**
맵에서 노드를 고를 때마다 스냅숏을 남기고, 런이 끝나면 라벨을 붙입니다.
- 노드: 그 뒤 실제로 지나간 노드 중 죽은 노드 = 1, 살아서 지나간 노드 = 0, 죽은 뒤 노드 = 라벨 없음
- 막: 실제로 도착한 미래 막만 라벨이 붙습니다 (그 막에서 죽었으면 1).

**모든 계산은 log 공간에서 합니다.** 에이전트가 약할 때는 모든 상태의 P(런 클리어)가 0에 가깝습니다. 확률을 그대로 곱하면 값이 0으로 뭉개져서 비교가 안 됩니다(실제로 첫 학습에서 GRPO 신호가 사라졌습니다). log로 계산하면 "클리어 확률이 e배 높다" 같은 비율 차이가 남습니다.

## 전투 에이전트: GRPO

같은 전투를 G번 분기합니다(분기마다 셔플과 적 난수를 다시 뽑아서, 숨은 정보를 엿보지 못하게 합니다). 정책으로 끝까지 굴린 다음, 전투 직후 런 상태를 평가기로 채점합니다.

- 점수: `log P(런 클리어)`. 전투에서 죽으면 그룹의 최저 점수보다 3 nat 낮은 값을 줍니다.
- 이점(advantage):
  - `zscore`(기본): `(점수 - 그룹 평균) / 그룹 표준편차`. 표준 GRPO입니다.
  - `best_vs_rest`: 1등 +1, 나머지 -1/(G-1). "가장 높게 평가된 방식에 보상, 나머지에 벌"을 그대로 옮긴 것입니다.
- 업데이트는 PPO식 clipped ratio를 씁니다. 시행 안에서 스텝 평균, 그다음 시행들끼리 평균을 냅니다.
- 평가기가 스냅숏 200개를 모으기 전(워밍업)에는 남은 HP 비율로 채점합니다.

critic(가치망) 없이 그룹 평균이 기준선 역할을 하는 것이 GRPO의 핵심입니다. 같은 상태에서 여러 번 굴려 서로 비교하는 방식은 RLOO(REINFORCE Leave-One-Out)와도 가깝습니다.

### 전투만 따로 학습 (`scripts/train_combat.py`)

런 안에서만 배우면 전투 에이전트는 카드 선택이 고른 카드만 많이 보게 됩니다. 그러면 "강한 카드"가 아니라 "전투 에이전트가 이미 잘 쓰는 카드"가 계속 뽑히는 쪽으로 굳습니다. 그래서 런 없이 무작위 상황을 만들어 모든 카드, 모든 적과 싸우게 합니다(`sts2ai/train/combat_train.py`).

- **시나리오:** 무작위 덱(`--deck-min 10`~`--deck-max 50`장, 카드 풀 전체에서 고르고 `--upgrade-prob 0.2`로 강화), 무작위 포션(0~9개)을 줍니다. 최대 HP는 1~125(`--max-hp-min`, `--max-hp-max`)에서, HP는 1~최대 HP에서 뽑습니다. 유물은 없습니다.
- **상대:** 반복마다 두 가지 방식으로 고릅니다.
  - **고루:** (조우, 막) 전체(토이에서는 6 × 3 = 18개)를 `--rounds-per-iter`바퀴 돕니다. 모든 적과 반드시 싸웁니다.
  - **집중:** `--focus-per-iter`개(기본 18)를 '만났을 때 사망률'에 비례해 더 뽑습니다.
  - 사망률은 학습 중 GRPO 시행의 결과로 (조우, 막)마다 지수이동평균을 냅니다(`--death-rate-ema 0.3`, 처음엔 모두 0.5).
  - 뽑을 확률은 `max(사망률, --focus-floor)^--focus-power`에 비례합니다. 그래서 사망률이 0이 된 적도 가끔은 집중 쪽에서 뽑힙니다.
  - 로그에 사망률이 높은 조우@막 상위 4개가 나옵니다.
- **GRPO 그룹:** 같은 시나리오, **같은 시드**로 G번 싸웁니다. 셔플과 적 난수가 같으므로 시행끼리의 차이는 정책의 선택에서만 나옵니다.
- **점수:** 이긴 시행은 항상 진 시행보다 높습니다.
  - 이김: `2 + clip(q/125, -1, 1)`, `q = ΔHP + 1·Δ최대HP` → 범위 [1, 3]. 포션은 마음껏 씁니다(쓴 만큼 점수를 깎지 않음).
  - 짐: `−1 + 적에게 준 피해 비율` → 범위 [−1, 0]. 그룹 전체가 져도 학습 신호가 남습니다. `--no-loss-ranking`이면 −1로 고정합니다.
  - 가중치는 `--max-hp-weight`로 바꿉니다. 포션을 아껴 쓰게 하려면 `--potion-cost-hp`(포션 1개 = HP 몇, 기본 0)를 0보다 크게 줍니다. 토이에서 10으로 해 보니 포션을 아끼다 HP를 더 잃었고, 승률도 오르지 않았습니다.
- **런 맥락:** 전투만 보므로 RUN 토큰은 넣지 않습니다. 전투 관측의 전역 토큰에 HP 비율 말고 절대 HP(`F_HP_ABS`)와 최대 HP(`F_MAX_HP`)를 추가했습니다. 그래야 HP 1과 HP 125를 구분합니다.
- **검증:** `--eval-every`마다 고정된 검증 시나리오(`--eval-rounds` 바퀴)를 greedy로 평가하고, 조우·막별 승률을 출력합니다.

```bash
.venv/bin/python scripts/train_combat.py --iterations 200 --out-dir runs/combat
```
```bash
.venv/bin/python scripts/train.py --env toy --init-policy runs/combat/combat_policy.pt   # 런 학습 (전투 정책 고정)
```

**런 학습에서는 전투 에이전트를 고정합니다**(`freeze_policy`, 기본값). 런 학습은 다른 모델(평가기)을 학습하는 단계이기 때문입니다.
- `--init-policy`로 불러온 전투 정책의 가중치를 바꾸지 않습니다. GRPO 분기와 업데이트를 아예 하지 않아서 반복도 훨씬 빨라집니다.
- 평가기는 공유 임베딩의 복사본을 따로 가집니다. 평가기를 학습해도 전투 정책이 보는 표현이 바뀌지 않습니다. 복사본은 전투 정책의 임베딩에서 출발합니다.
- 전투만 따로 학습한 정책은 RUN 토큰을 본 적이 없으므로 넣지 않습니다. 이 설정은 `agent.pt`에 같이 저장되어 `scripts/evaluate.py`에서도 유지됩니다.
- `--init-policy` 없이 고정하려고 하면 에러가 납니다. 예전처럼 런 안에서 전투 정책도 같이 학습하려면 `--no-freeze-policy`를 주세요.

#### 실제 StS2 전투로 학습 (zhiyue/sts2-rl-agent)

[zhiyue/sts2-rl-agent](https://github.com/zhiyue/sts2-rl-agent)는 StS2 전투를 순수 Python으로 다시 구현한 시뮬레이터입니다. 이 저장소에는 그 코드를 넣지 않고, 따로 받아 둔 체크아웃을 경로로 불러 씁니다. 그 저장소에는 라이선스 파일이 없고, 게임을 디컴파일한 소스(`decompiled/`)가 들어 있기 때문입니다. 쓸지는 각자 판단하세요.

```bash
git clone https://github.com/zhiyue/sts2-rl-agent ../sts2-rl-agent && uv pip install gymnasium
```
```bash
.venv/bin/python scripts/train_combat.py --env zhiyue --zhiyue-path ../sts2-rl-agent --out-dir runs/combat_sts2
```
```bash
ZHIYUE_PATH=../sts2-rl-agent .venv/bin/python -m pytest -q tests   # 어댑터 테스트 (경로가 없으면 건너뜀)
```

- **대상:** 아이언클래드, 카드 풀 87장, 포션 64종, 1~4막 조우 80개(약한/일반/엘리트/보스), 몬스터 121종. 포식이나 포션 슬롯 9칸 같은 실제 메커니즘이 그대로 들어옵니다.
- **관측:** 그 시뮬레이터의 전투 상태를 우리 토큰으로 바꿉니다. 손패, 턴 종료, 포션 칸, 더미 3개, 플레이어·적 파워, 적과 의도(한 행동에 여러 개면 전부)가 들어갑니다. 카드 효과로 선택 화면이 뜨면 선택지 카드와 확인/건너뛰기 토큰이 행동이 됩니다(최대 31개).
- **합법 행동:** 그 시뮬레이터의 `get_action_mask`를 그대로 옮깁니다. 무작위 타깃이나 아군 타깃처럼 우리가 타깃을 고르지 않는 행동은 합법인 첫 타깃으로 보냅니다.
- **확인한 것:** 조우 80개 전부에서 무작위 시나리오와 무작위 행동으로 에러가 없습니다. 같은 시드의 전투를 번갈아 진행해도 똑같이 흘러갑니다(나란히 진행하는 GRPO 그룹의 전제). 포식으로 최대 HP가 오릅니다. 포션 9칸이 됩니다. 시뮬레이터만 돌리면 무작위 행동 기준 초당 약 160 전투(2,500 스텝)입니다.
- **그 시뮬레이터의 특이점:**
  - 카드 메타데이터를 그 저장소의 `docs/`와 `decompiled/`에서 현재 폴더 기준 경로로 읽습니다. 그래서 어댑터가 초기화할 때 그 폴더로 들어가 캐시를 한 번 채웁니다.
  - import 순서에 따라 순환 import 에러가 나서, `gym_env.combat_env`를 먼저 import합니다.
- **한계:**
  - 아이언클래드만 연결했고, 유물은 넣지 않습니다.
  - 정확도는 그 저장소의 주장이고, 이 저장소에서는 실제 게임과 대조하지 않았습니다.
  - 토이에서 학습한 체크포인트와는 어휘 크기가 달라서 호환되지 않습니다.

## 실행

```bash
uv venv .venv && uv pip install -r requirements.txt
```
```bash
.venv/bin/python -m pytest -q tests
```
```bash
.venv/bin/python scripts/train_combat.py --iterations 200 --out-dir runs/combat   # 1) 전투 에이전트
```
```bash
.venv/bin/python scripts/train.py --env toy --iterations 30 --init-policy runs/combat/combat_policy.pt   # 2) 평가기 (전투 고정)
```
```bash
.venv/bin/python scripts/evaluate.py --ckpt runs/latest/agent.pt --runs 50
```
```bash
.venv/bin/python scripts/calibration.py --ckpt runs/latest/agent.pt --runs 50 --sample
```

학습 로그 열:
- `win_rate`, `mean_floor`, `mean_act`: 학습 중 런 성적
- `fork_combat_win`: GRPO 분기 전투의 승률
- `scorer`: `hp-warmup` 또는 `evaluator`
- `node_bce`, `act_bce`: 평가기 손실
- `grpo_trajs`: 학습 신호가 있었던 시행 수

### 실제 시뮬레이터로 돌리기

```bash
git clone https://github.com/r33hab/sts2.git ../sts2
```
```bash
cd ../sts2 && bash scripts/build.sh && uv sync && cd -
```
```bash
.venv/bin/python scripts/train.py --env sts2 --sts2-repo ../sts2 --d-model 128
```

`sts2_adapter.py`는 시뮬레이터 소스(2026-10-05 커밋 기준)를 읽고 작성했지만 아직 실행해 보지 않았습니다. 확인이 필요한 가정은 다음과 같습니다.
1. 맵 선택지 좌표(`run_state_list` 7번)의 `(x, y)`가 `map_graph()`의 `(col, row)`와 같은 좌표계다.
2. 전투 중 `run_action_mask`의 앞 32칸이 전투 행동 번호와 같다.
3. 시뮬레이터가 2·3막(Hive, Glory)을 얼마나 구현했는지는 확인하지 않았다. 문서상으로는 1막 위주로 검증돼 있다.

### 배치 롤아웃과 GPU

학습 루프는 런 `--parallel-runs`판(기본값 0은 `--runs-per-iter` 전부)을 나란히 진행합니다. 그 런들의 GRPO 분기 전투와 진짜 전투가 매 스텝 요청하는 행동을 모아서 정책 forward 한 번으로 처리합니다(`grpo.drive`). 한 스텝의 배치는 최대 `parallel_runs × group_size` 정도입니다. 전투 코드는 정책이 필요할 때 `yield`로 행동을 요청하는 제너레이터로 되어 있습니다(`play_steps`, `rollout_group_steps`, `play_run_steps`).

```bash
.venv/bin/python scripts/train.py --env toy --init-policy runs/combat/combat_policy.pt --iterations 30 --device auto   # GPU 가 있으면 cuda
```

CPU 4코어에서 잰 수치입니다(토이, 기본 설정, `--warmup-snapshots 50`).

| | 반복 1회 (워밍업) | 반복 1회 (평가기 사용) |
|---|---|---|
| 이전 코드 (하나씩) | 17초 | 34초 |
| `--parallel-runs 1` (그룹 안에서만 배치) | 11~15초 | 24초 |
| `--parallel-runs 8` | 9초 | 12초 |

위 표는 런 안에서 전투 정책도 학습하던 설정(`--no-freeze-policy`)으로 잰 것입니다. 전투 정책을 고정하면 런 학습에는 GRPO 분기가 없고, 나란히 진행하는 런들의 진짜 전투만 한 배치로 묶입니다. 전투만 따로 학습할 때(`train_combat.py`)는 시나리오 전부 × 그룹 크기(기본 36 × 8)가 한 배치입니다.

`scripts/benchmark.py`에서 분기 전투 16개를 같이 굴리면 초당 23개에서 121개가 됩니다. GPU 수치는 아직 재지 않았습니다. 맵 노드 선택, 전투 시작 때의 `run_vec`, 기대값 선택기는 아직 런마다 따로 계산합니다.

### 클라우드에서 돌리기 (이어서 학습, 끝나면 인스턴스 종료)

학습은 반복이 끝날 때마다 `<out-dir>/checkpoint.pt`에 전체 상태를 저장합니다. 저장되는 것은 가중치, 두 옵티마이저, 평가기 버퍼, 모든 난수 상태, 기록입니다. `--resume`을 주면 마지막으로 끝난 반복 다음부터 이어 갑니다. 끊겼다 이어도 끊기지 않고 돈 것과 같은 결과가 나오는지 테스트로 확인합니다(`test_resume_matches_uninterrupted_training`). 바꿀 수 있는 설정은 `--iterations`(목표 반복 수 늘리기)와 `--torch-threads`뿐이고, 나머지가 다르면 거부합니다.

```bash
.venv/bin/python scripts/train.py --env toy --init-policy runs/combat/combat_policy.pt --iterations 200 --out-dir runs/a --resume
```

`scripts/cloud_train.sh`는 위 명령을 감싸서 다음을 합니다.
- 항상 `--resume`으로 실행합니다. 스팟 회수로 꺼졌다가 다시 켜지면 이어서 학습합니다.
- 끝나면 `DONE`, 실패하면 `FAILED` 표시 파일을 남깁니다. 둘 중 하나가 있으면 부팅 때 다시 학습하지 않습니다.
- `--shutdown`을 줬을 때만, 끝난 뒤(성공이든 실패든) 1분 뒤에 `shutdown -h`로 인스턴스를 끕니다. 로컬 PC에서는 빼고 쓰세요.
- 시그널로 죽은 경우(스팟 회수, 시스템 종료)는 실패로 치지 않습니다.

```bash
scripts/cloud_train.sh --out-dir runs/a --shutdown -- --env toy --init-policy runs/combat/combat_policy.pt --iterations 200
```

부팅할 때 자동으로 실행하려면 systemd 서비스로 등록합니다. 경로는 환경에 맞게 바꾸세요.

```ini
# /etc/systemd/system/sts2-train.service
[Unit]
Description=sts2_AI training
After=network-online.target
Wants=network-online.target

[Service]
User=root
WorkingDirectory=/opt/sts2_AI
ExecStart=/opt/sts2_AI/scripts/cloud_train.sh --out-dir /data/runs/a --shutdown -- --env toy --init-policy /data/runs/combat/combat_policy.pt --iterations 200
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload && sudo systemctl enable --now sts2-train
```

인스턴스를 끄면 과금이 멈추는지는 서비스마다 다릅니다. 쓰기 전에 각 서비스의 문서로 확인하세요.
- **AWS EC2, GCP Compute Engine:** OS에서 끄면 인스턴스가 '정지' 상태가 되고 CPU·GPU 요금은 멈춥니다. 디스크 요금은 계속 나옵니다. 인스턴스 종료 동작이 'terminate'로 설정돼 있으면 루트 디스크가 지워질 수 있으니, `--out-dir`은 따로 붙인 볼륨에 두세요.
- **스팟 인스턴스:** 회수되면 그 인스턴스의 디스크가 사라질 수 있습니다. 이어서 학습하려면 체크포인트가 영구 볼륨에 있어야 합니다(예: AWS의 영구 스팟 요청 + 중단 동작 'stop').
- **Hetzner Cloud:** 꺼져 있어도 서버 요금이 나옵니다. 요금을 멈추려면 서버를 삭제해야 합니다.
- **vast.ai, RunPod 같은 컨테이너형 GPU 대여:** 컨테이너 안에서 `shutdown`해도 과금이 멈추지 않습니다. `SHUTDOWN_CMD`에 그 서비스의 정지 명령을 넣으세요. 예: `SHUTDOWN_CMD='vastai stop instance $CONTAINER_ID'`.

## 토이 게임 결과

설정: `--iterations 30 --runs-per-iter 8 --group-size 8`, d_model 64, CPU 기준 약 25분. 평가는 학습에 쓰지 않은 시드 50판으로 했습니다. 각 설정을 시드 하나로 한 번씩만 학습했으므로, 버전 사이의 작은 차이는 우연일 수 있습니다.

| 버전 | 클리어율 (greedy) | 평균 도달 층 (greedy) | 클리어율 (샘플링) | 노드 평가기 BCE | 미래 막 평가기 BCE |
|---|---|---|---|---|---|
| 학습 전 | 0% | 6.1 (샘플링) | 0% | – | – |
| v1: 정규화 없음 | 14% | 23.7 | 4% | 0.320 (상수 예측 0.227) | 0.797 (상수 0.660) |
| v2: 검증 + 되돌리기 | **40%** | 24.7 | 14% | **0.158** (상수 0.216) | 1.241 (상수 0.680) |
| v3: 첫 에폭은 항상 반영 (현재 코드) | 14% | 23.7 | 16% | 0.277 (상수 0.227) | **0.578** (상수 0.663) |

(토이 게임은 3막 × 9층 = 27층입니다. BCE는 낮을수록 좋고, "상수"는 항상 평균 사망률만 말하는 예측기입니다.)

정리하면 이렇습니다.
- **전투 에이전트는 잘 배웁니다.** GRPO만으로 모든 버전이 학습 전보다 크게 좋아졌습니다.
- **평가기는 아직 불안정합니다.**
  - v1: 같은 버퍼를 계속 다시 학습해서 과적합했고, 확신이 지나쳤습니다("97% 죽음" → 실제 46%).
  - v2: 되돌리기 때문에 9번째 반복부터 평가기가 사실상 얼어 버렸습니다. 노드 평가기는 오히려 괜찮았지만, 미래 막 평가기는 초반 값("2막 = 90% 사망")에 고정됐습니다.
  - v3: 미래 막 평가기는 고쳐졌지만 노드 평가기가 다시 과적합합니다.
### v4: 맵을 보는 전투 정책 + 선택 화면이 있는 토이 (현재 코드)

v3와 같은 설정에 RUN 토큰(전투 정책이 맵을 봄)과 새 선택 화면(무작위 변환, 무작위 강화 유물, 상점 제거)을 더했습니다. CPU 1스레드로 36분 걸렸습니다. 전투 정책은 v3와 비슷하게 배웠지만, RL 루프 안의 평가기는 학습 손실 0.016에 검증 손실 1.43으로 과적합이 더 심했습니다.

### 실험 1: 데이터를 늘리면 평가기가 좋아지나 (`scripts/data_scaling.py`)

v4의 전투 정책·맵 선택을 고정하고 런을 모은 뒤, 학습 런 수만 바꿔 평가기를 처음부터 학습했습니다(조기 종료). 테스트는 공통 150판이고, 사건 단위로 쟀습니다.

| 학습 런 | 노드 테스트 BCE (상수 0.173) | ECE | 예측 0.7 이상 → 실제 | 미래 막 테스트 BCE (상수 0.693) |
|---|---|---|---|---|
| 25 | 0.144 | 0.015 | 해당 없음 | 0.625 |
| 50 | 0.082 | 0.010 | 0.78 → 0.70 | 0.634 |
| 100 | 0.109 | 0.022 | 0.92 → 0.70 | 0.589 |
| 200 | 0.072 | 0.010 | 0.87 → 0.78 | 0.584 |
| 400 | 0.074 | 0.011 | 0.84 → 0.80 | 0.590 |
| 800 | 0.074 | 0.005 | 0.86 → 0.83 | 0.575 |

- 약 200판까지는 데이터가 도움이 되고, 그 뒤로는 거의 늘지 않습니다. 확신 과잉은 데이터가 늘수록 줄어듭니다.
- 고정된 데이터로 따로 학습한 평가기는 잘 보정됩니다. RL 루프에서 평가기가 나빴던 주원인은 데이터 양보다는 "정책이 바뀌는 중에 같은 버퍼를 반복 학습하는 것"으로 보입니다.

### 실험 2: 기대값 선택기는 도움이 되나 (`scripts/chooser_experiment.py`)

같은 테스트 시드 100판에서, 전투·맵 밖 선택 방식만 바꿔 비교했습니다.

| 전투·맵 밖 선택 | 클리어율 (± 표준오차) | 평균 도달 층 |
|---|---|---|
| 규칙 정책 | 23% ± 4.2 | 22.4 |
| 선택기 + RL 평가기 | 24% ± 4.3 | 21.0 |
| 선택기 + 따로 학습한 평가기 (400판) | 19% ± 3.9 | 19.1 |

셋 다 오차 범위 안이라, 선택기가 도움이 된다는 증거는 없습니다. 50판씩 처음 비교했을 때는 30% 대 16%였는데, 이는 표본이 적어서 생긴 차이였습니다.

### 실험 3: 평가기가 카드 가치를 구분하나

같은 상태에서 카드를 한 장씩 빼 보고 log V 변화를 쟀습니다(상태 265개).
- RL 평가기는 "강화 안 된 카드는 빼면 이득(+0.04~+0.10), 강화된 카드는 빼면 손해(−0.09~−0.46)"만 배웠습니다. 카드 종류는 구분하지 못해서, Cleave나 Heavy Blade를 빼는 게 Strike보다 낫다고 판단합니다. 가장 빼고 싶은 카드가 타격·수비인 상태는 21%뿐이었습니다.
- 따로 학습한 평가기는 변화량이 ±0.005 정도로, 카드 한 장에 사실상 반응하지 않습니다.
- 덱 인코더에 보조 특징으로 넣은 "강화 비율"에 기대는 지름길을 배운 것으로 보입니다. 사망 확률을 잘 맞히는 것과 "이 카드를 빼면 어떻게 되나"를 맞히는 것은 다른 문제입니다. 노드 사망 라벨만으로는 카드 한 장의 효과가 너무 약한 신호입니다.

### 다음 단계

- 덱 인코더 사전학습: 전투 에이전트로 "이 덱으로 대표 전투를 이길 확률·잃는 HP"를 시뮬레이션해서 라벨을 만듭니다. 카드 한 장을 넣고 뺀 쌍을 비교하면 카드 가치가 직접 신호가 됩니다.
- 덱 통계 특징(강화 비율 등)을 빼고 다시 확인합니다.
- RL 루프의 평가기 학습을 "최근 정책의 데이터만, 반복마다 처음부터 또는 적은 에폭"으로 바꿉니다.
- 분기 전투를 여러 프로세스로 나눠 돌려 속도를 높입니다.

## 전투·맵 밖 선택

카드 보상, 상점, 휴식, 이벤트, 덱에서 카드 고르기는 두 가지 방식 중 하나로 고릅니다(`--chooser rules|ev`).
- `rules`: 간단한 규칙 (기본값)
- `ev`: 기대값 선택기 (`sts2ai/agents/chooser.py`). 숨은 난수를 다시 뽑은 복사본으로 선택지마다 M번 굴려 `log E[V]`를 비교합니다. 다음 화면이 또 선택이면 이어서 계산합니다. 위 실험 2처럼 아직 규칙보다 낫다는 증거는 없습니다.

## 출처

전투 토큰 규약(토큰 종류, 특징 채널 번호, "토큰 1 + i = 행동 i")은 r33hab/sts2의 `rl/combat_tokens.py`를 따랐고, 어댑터는 그 모듈을 그대로 불러 씁니다.
