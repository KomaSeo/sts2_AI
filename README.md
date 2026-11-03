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
| `sts2ai/envs/toy.py` | 검증용 토이 게임 |
| `sts2ai/envs/sts2_adapter.py` | r33hab/sts2 어댑터 (**실행 검증 전**) |

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

## 실행

```bash
uv venv .venv && uv pip install -r requirements.txt
```
```bash
.venv/bin/python -m pytest -q tests
```
```bash
.venv/bin/python scripts/train.py --env toy --iterations 30
```
```bash
.venv/bin/python scripts/evaluate.py --ckpt runs/latest/agent.pt --runs 50
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

## 범위 밖 (규칙 정책으로 처리)

카드 보상, 상점, 휴식, 이벤트는 간단한 규칙으로 고릅니다. 다음 단계 후보:
- 같은 평가기로 카드 보상 고르기: 후보마다 덱에 넣은 상태를 만들어 log V가 가장 높은 것 선택
- 전투 시작 상태를 저장해 두고 GRPO를 오프라인으로 반복
- 평가기 보정(calibration) 측정: 예측한 사망 확률과 실제 사망률 비교

## 출처

전투 토큰 규약(토큰 종류, 특징 채널 번호, "토큰 1 + i = 행동 i")은 r33hab/sts2의 `rl/combat_tokens.py`를 따랐고, 어댑터는 그 모듈을 그대로 불러 씁니다.
