# AWS 스팟 GPU에서 전투 에이전트 학습하기

전투 에이전트만 따로 학습하는 실험(`scripts/train_combat.py`, zhiyue StS2 전투)을 AWS EC2 스팟 GPU 서버에서 돌리는 순서입니다. 처음 한 번은 1~2시간쯤 걸리고, 그다음부터는 4~7단계만 반복하면 됩니다.

필요한 파일:
- 이 저장소 (`ccr-efec8c2b-ct4hfn` 브랜치 또는 이 브랜치를 합친 main)
- 체크포인트 묶음 `sts2_checkpoints.tar.gz` (이어서 학습할 때만)

## 1. 계정 준비 (처음 한 번)

1. **예산 알림부터 겁니다.** Billing → Budgets → 월 예산(예: $30)을 만들고, 80%·100%에서 이메일을 받게 합니다. 켜 둔 채 잊은 서버가 가장 큰 비용 위험입니다.
2. **GPU 스팟 한도를 늘립니다.** 새 계정은 GPU 스팟 한도가 0이라 서버를 아예 못 띄웁니다.
   - Service Quotas → Amazon EC2 → **All G and VT Spot Instance Requests**
   - 값을 **8** 이상으로 요청합니다(vCPU 수 기준, g4dn.2xlarge 한 대 = 8).
   - 승인에 몇 시간에서 하루쯤 걸릴 수 있으니 미리 해 둡니다. 서버를 띄울 리전(예: us-east-1)에서 요청해야 합니다.
3. **키 페어를 만듭니다.** EC2 → Key Pairs → Create. `.pem` 파일을 받아 `chmod 400`.

## 2. 서버 고르기

| 인스턴스 | GPU | vCPU | 메모리 | 메모 |
|---|---|---|---|---|
| **g4dn.2xlarge** (추천) | T4 16GB | 8 | 32GB | 실험 5개를 동시에 돌릴 수 있는 코어 수 |
| g4dn.xlarge | T4 16GB | 4 | 16GB | 실험 1~3개만 돌릴 때. 2026년 11월 스팟 대략 $0.16~0.32/시간 |
| g5.2xlarge | A10G 24GB | 8 | 32GB | 더 빠른 GPU. 이 모델 크기에서는 T4로 충분할 가능성이 큼 |

- 실험 하나가 CPU 1코어를 씁니다(시뮬레이터가 순수 Python). GPU는 모든 실험이 나눠 씁니다.
- 스팟 가격은 리전과 가용 영역마다 다르고 수시로 바뀝니다. 띄우기 전에 EC2 콘솔의 Spot Requests → Pricing history에서 확인하세요. 서울(ap-northeast-2)보다 미국 동부(us-east-1)가 보통 쌉니다. 이 작업은 응답 속도가 상관없으니 싼 리전을 쓰면 됩니다.

## 3. 서버 띄우기

EC2 → Launch instance:

- **AMI:** NVIDIA 드라이버가 깔린 Ubuntu 딥러닝 AMI. 검색창에 `Deep Learning Base` 를 치고 "OSS Nvidia Driver GPU AMI (Ubuntu)" 계열을 고릅니다. Python 버전은 상관없습니다(설치 스크립트가 3.12를 따로 깝니다).
- **인스턴스 유형:** g4dn.2xlarge
- **키 페어:** 1단계에서 만든 것
- **네트워크:** 보안 그룹에서 SSH(22)를 **내 IP**에서만 허용
- **스토리지:** AMI가 요구하는 최소 크기 이상의 gp3 (학습 결과는 실험당 수 MB라 여유는 충분)
- **고급 세부 정보:**
  - 구매 옵션: **스팟 인스턴스**
  - 요청 유형: **영구(persistent)**, 중단 동작: **중지(stop)**. 스팟이 회수돼도 디스크가 남고, 자리가 나면 다시 켜져 이어서 학습합니다.
  - 종료 동작(Shutdown behavior): **중지(stop)**. 학습이 끝나 서버가 꺼질 때 디스크를 지우지 않습니다.

## 4. 설치 (서버마다 처음 한 번)

```bash
ssh -i my-key.pem ubuntu@<서버 공개 IP>
```
```bash
git clone https://github.com/KomaSeo/sts2_AI.git sts2_AI && cd sts2_AI && git checkout ccr-efec8c2b-ct4hfn
```
저장소가 비공개라면 clone할 때 비밀번호 자리에 GitHub 개인 액세스 토큰을 넣습니다(Settings → Developer settings → Fine-grained token, 이 저장소의 Contents 읽기 권한만).

```bash
bash scripts/cloud_setup.sh --bench
```
uv와 Python 3.12, CUDA용 torch를 설치하고, zhiyue 시뮬레이터를 저장소 옆(`../sts2-rl-agent`)에 검증한 커밋으로 받고, 테스트와 GPU 시험 학습을 돌립니다. `--bench`는 기본 설정으로 반복 1회 시간을 재서 마지막에 `sec=...`로 보여 줍니다. **이 숫자로 5단계의 반복 수를 정합니다.**

이어서 학습하려면 체크포인트를 올려서 풉니다(내 컴퓨터에서):
```bash
scp -i my-key.pem sts2_checkpoints.tar.gz ubuntu@<IP>:sts2_AI/ && ssh -i my-key.pem ubuntu@<IP> 'cd sts2_AI && tar xzf sts2_checkpoints.tar.gz'
```

## 5. 실험 고르기

계획 파일은 `experiments/`에 있습니다.

| 파일 | 내용 |
|---|---|
| `experiments/size_lr.txt` | 모델 크기 3종(22만·81만·388만) + 학습률 3배 + 업데이트 2배, 5개를 처음부터 동시에 학습. 검증 400판 |
| `experiments/continue_relics.txt` | 이 세션의 유물 학습(반복 18)을 60번까지 이어서 학습 |

반복 수는 `@common` 줄의 `--iterations`를 고칩니다. 대략 `반복 수 × bench 시간 × (동시 실험이 GPU·CPU를 나눠 쓰는 만큼의 여유)`가 걸립니다. 처음에는 짧게(예: 10번) 돌려 실제 속도를 보고 늘리는 것을 권합니다. 같은 폴더에 다시 돌리면 이어서 학습하니 손해가 없습니다.

## 6. 실행 (끝나면 자동으로 꺼지게)

서버가 스팟 회수 뒤 다시 켜져도 자동으로 이어서 돌도록 systemd 서비스로 등록합니다.

```bash
sudo tee /etc/systemd/system/sts2-sweep.service >/dev/null <<'UNIT'
[Unit]
Description=sts2_AI experiment sweep
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/sts2_AI
ExecStart=/home/ubuntu/sts2_AI/scripts/cloud_sweep.sh --plan experiments/size_lr.txt --runs-dir runs/size_lr --shutdown
KillMode=control-group
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
UNIT
```
```bash
sudo systemctl daemon-reload && sudo systemctl enable --now sts2-sweep
```

- 모든 실험이 끝나면(완료나 실패) 1분 뒤 서버가 꺼집니다. SSH로 붙어 있다면 `sudo shutdown -c`로 취소할 수 있습니다.
- 스팟 회수로 중간에 꺼지면 서버를 끄는 명령을 내리지 않고, 다시 켜졌을 때 끝나지 않은 실험만 이어서 돕니다.
- 실패한 실험은 `runs/size_lr/<이름>/FAILED`가 남고, 다시 돌려도 건너뜁니다. 원인을 고친 뒤 그 파일을 지우면 다시 돕니다.
- 다른 계획을 돌리려면 서비스 파일의 `--plan`과 `--runs-dir`를 바꾸고 `sudo systemctl daemon-reload && sudo systemctl restart sts2-sweep`.

## 7. 지켜보기와 결과 받기

```bash
tail -f runs/size_lr/sweep.log                                          # 실험 시작·끝
grep -h "eval_win_rate" runs/size_lr/*/train.log | grep -oE "^iter=[0-9]+|eval_win_rate=[0-9.]+" | paste - -   # 검증 승률
```
결과를 내 컴퓨터로 받기:
```bash
scp -i my-key.pem -r ubuntu@<IP>:sts2_AI/runs/size_lr ./
```
`combat_history.json`(반복별 기록)과 `combat_policy.pt`(가중치)가 있으면 학습 곡선 비교와 전투 리플레이(`scripts/show_combat.py`)를 만들 수 있습니다. 서버가 꺼져 있어도 디스크는 남아 있으니, 콘솔에서 다시 시작해 받으면 됩니다(서비스가 다시 돌지만 모두 DONE이라 바로 다시 꺼집니다. 받는 동안 꺼지지 않게 하려면 먼저 `sudo systemctl disable --now sts2-sweep`).

## 8. 정리 (다 끝났을 때)

순서가 중요합니다.

1. **스팟 요청을 취소합니다.** EC2 → Spot Requests → 해당 요청 → Cancel. 영구 요청을 남겨 두면 인스턴스를 지워도 새로 띄울 수 있습니다.
2. **인스턴스를 종료(Terminate)합니다.**
3. **볼륨과 스냅샷을 지웁니다.** EC2 → Volumes, Snapshots에서 남은 것이 없는지 확인합니다. 꺼진 서버도 디스크 요금은 계속 나옵니다(gp3 기준 대략 GB당 월 $0.08).
4. 다음 날 Billing에서 요금이 멈췄는지 확인합니다.
