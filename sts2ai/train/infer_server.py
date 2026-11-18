"""전투 정책 추론 서버: 수집 워커들의 정책 호출을 한 프로세스(GPU)가 모아서 한 배치로 계산한다.

수집 시간의 대부분은 전투 정책 호출이다. 워커 하나가 부르는 배치는 작아서(분기 8개 묶어도 8) CPU 로는
관측당 1ms 쯤 들지만, GPU 는 배치 100 근처에서도 호출 한 번 시간이 거의 같아서 여러 워커 요청을 합치면
관측당 시간이 훨씬 작아진다.

- 요청: (워커 번호, 관측 목록, run_vec numpy [B,d] 또는 None). 모든 워커가 요청 큐 하나에 넣는다.
- 서버는 첫 요청을 받은 뒤 max_wait 초 동안(또는 max_batch 행이 찰 때까지) 더 받아서 한 번에 계산한다.
- 응답: 행동 logit [B,A], 행동마다의 타깃 logit [B,A,E], 타깃 필요 여부 [B,A], 적 마스크 [B,E].
  행동·타깃 샘플링은 워커가 자기 torch 난수로 한다 (서버가 뽑으면 어느 요청과 묶였는지에 따라 결과가 바뀐다).
- 가중치: 학습 루프가 반복마다 제어 파이프로 보내고, 서버가 다 읽었다는 답을 받은 뒤 다음 수집을 시작한다.
"""

from __future__ import annotations

import io
import multiprocessing as mp
import queue
import time

import numpy as np
import torch

from ..batching import collate_combat, to_device
from ..schema import MAX_ACTIONS

NEG = -1e9


@torch.no_grad()
def policy_dists(policy, batch: dict, run_vec: torch.Tensor | None):
    """행동 logit [B,A] 와, 행동 a 를 골랐을 때의 타깃 logit 을 모든 a 에 대해 [B,A,E]."""
    out = policy(batch, run_vec)
    h = out.tokens
    B, T, d = h.shape
    idx = (1 + torch.arange(MAX_ACTIONS, device=h.device)).clamp(max=T - 1)  # target_logits 와 같은 clamp
    q = policy.tq(h[:, idx])  # [B,A,d]
    b = torch.arange(B, device=h.device)
    k = policy.tk(h[b.unsqueeze(1), batch["enemy_tok"]])  # [B,E,d]
    tl = torch.einsum("bad,bed->bae", q, k) / d ** 0.5
    tl = tl.masked_fill(~batch["enemy_mask"].unsqueeze(1), NEG)
    return out.action_logits, tl


def sample_from(action_logits, target_logits, needs_target, enemy_mask, greedy: bool):
    """CombatPolicy.act 와 같은 방식으로 (행동, 타깃 인덱스, log 확률) 을 뽑는다."""
    dist = torch.distributions.Categorical(logits=action_logits)
    a = action_logits.argmax(-1) if greedy else dist.sample()
    lp = dist.log_prob(a)
    t = torch.full_like(a, -1)
    needs = needs_target.gather(1, a.unsqueeze(1)).squeeze(1) & enemy_mask.any(-1)
    if needs.any():
        tl = target_logits[torch.arange(a.shape[0]), a]  # [B,E]
        td = torch.distributions.Categorical(logits=tl)
        tt = tl.argmax(-1) if greedy else td.sample()
        t = torch.where(needs, tt, t)
        lp = lp + torch.where(needs, td.log_prob(tt), torch.zeros_like(lp))
    return a, t, lp


def _bucket(n: int, sizes) -> int:
    for s in sizes:
        if n <= s:
            return s
    return -1


B_SIZES = (8, 16, 32, 64, 128, 256, 512)
T_STEP = 16  # 토큰 수는 16 단위로 올려 채운다
E_SIZES = (1, 2, 3, 4, 6, 8)


class GraphedPolicy:
    """CUDA Graph 로 policy_dists 를 녹화해 두고 재생한다.

    이 모델은 작은 연산이 많아서, GPU 호출 시간의 대부분이 CPU 가 연산을 하나씩 GPU 에 보내는 시간이다
    (CPU 가 바쁘면 배치 64 한 번에 20ms 넘게 걸렸다). 녹화한 그래프는 한 번에 보내므로 약 1ms 로 줄었다.
    그래프는 입력 모양이 고정이어야 해서 (배치, 토큰 수, 적 수) 를 몇 단계 크기로 올려 채우고, 크기마다 처음 쓸 때 녹화한다.
    채운 행은 진짜 첫 행을 복사하고, 채운 토큰은 마스크로 가려서 결과에 영향을 주지 않는다.
    가중치를 바꿀 때는 같은 메모리에 복사하므로(load_state_dict) 녹화를 다시 할 필요가 없다.
    """

    def __init__(self, policy, dev):
        self.policy, self.dev = policy, dev
        self.graphs: dict = {}
        self.pool = None

    def _pad(self, batch: dict, B: int, T: int, E: int) -> dict:
        out = {}
        n = batch["types"].shape[0]
        for k, v in batch.items():
            shape = list(v.shape)
            shape[0] = B
            if k in ("types", "ids", "owners", "feats", "tok_mask", "enchants"):
                shape[1] = T
            elif k in ("enemy_tok", "enemy_mask"):
                shape[1] = E
            p = torch.zeros(shape, dtype=v.dtype)
            sl = tuple(slice(0, s) for s in v.shape)
            p[sl] = v
            if B > n:
                p[n:] = p[0]  # 빈 행은 첫 행 복사 (토큰이 하나도 없는 행은 attention 이 NaN 이 된다)
            out[k] = p
        return out

    @torch.no_grad()
    def __call__(self, batch: dict, run_vec: torch.Tensor | None):
        n, T0 = batch["types"].shape
        E0 = batch["enemy_tok"].shape[1]
        B, E = _bucket(n, B_SIZES), _bucket(E0, E_SIZES)
        T = -(-T0 // T_STEP) * T_STEP
        if B < 0 or E < 0:  # 너무 크면 그냥 계산
            return policy_dists(self.policy, to_device(batch, self.dev),
                                run_vec.to(self.dev) if run_vec is not None else None)
        key = (B, T, E, run_vec is not None)
        padded = self._pad(batch, B, T, E)
        if run_vec is not None:
            rv = torch.zeros(B, run_vec.shape[1])
            rv[:n] = run_vec
            rv[n:] = run_vec[0]
            padded["__rv"] = rv
        if key not in self.graphs:
            static = {k: v.to(self.dev) for k, v in padded.items()}
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    policy_dists(self.policy, static, static.get("__rv"))
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                outs = policy_dists(self.policy, static, static.get("__rv"))
            self.pool = g.pool()
            self.graphs[key] = (g, static, outs)
        g, static, outs = self.graphs[key]
        for k, v in padded.items():
            static[k].copy_(v, non_blocking=True)
        g.replay()
        al, tl = outs
        return al[:n], tl[:n, :, :E0]


def _serve(vocab, d_model: int, device: str, req_q, resp_qs, ctrl, max_batch: int, max_wait: float,
           policy_kwargs: dict | None = None) -> None:
    """policy_kwargs 가 있으면 전투 정책 하나만 그 크기(layers, heads, ff_mult)로 만들고, 가중치도 정책 것만 받는다
    (전투만 학습). 없으면 예전처럼 Agent 전체의 가중치를 받는다 (런 학습)."""
    from .loop import Agent  # noqa: PLC0415 (spawn 프로세스 안에서 불러온다)

    torch.set_num_threads(1)
    dev = torch.device(device)
    if policy_kwargs is not None:
        from ..agents import CombatPolicy  # noqa: PLC0415
        from ..encoders import SharedEmbeddings  # noqa: PLC0415

        agent = None
        policy = CombatPolicy(SharedEmbeddings(vocab, d_model), **policy_kwargs).to(dev).eval()
    else:
        agent = Agent(vocab, d_model)
        policy = agent.policy.to(dev).eval()
    graphed = None
    if dev.type == "cuda":
        torch.backends.mha.set_fastpath_enabled(False)  # Transformer 빠른 경로는 그래프로 녹화되지 않는다
        graphed = GraphedPolicy(policy, dev)
    stats = {"calls": 0, "rows": 0, "busy": 0.0}
    while True:
        if ctrl.poll():
            msg = ctrl.recv()
            if msg[0] == "stop":
                return
            if msg[0] == "weights":
                # 같은 메모리에 복사된다 (녹화한 그래프가 새 가중치를 그대로 쓴다)
                sd = torch.load(io.BytesIO(msg[1]), map_location="cpu")
                (policy if agent is None else agent).load_state_dict(sd)
                ctrl.send(("ok",))
            elif msg[0] == "stats":
                ctrl.send(("stats", dict(stats)))
                stats = {"calls": 0, "rows": 0, "busy": 0.0}
            continue
        try:
            first = req_q.get(timeout=0.01)
        except queue.Empty:
            continue
        reqs, rows = [first], len(first[1])
        deadline = time.perf_counter() + max_wait
        while rows < max_batch:
            try:
                r = req_q.get_nowait() if time.perf_counter() >= deadline else req_q.get(
                    timeout=max(0.0, deadline - time.perf_counter()))
            except queue.Empty:
                break
            reqs.append(r)
            rows += len(r[1])
        t0 = time.perf_counter()
        obs = [o for r in reqs for o in r[1]]
        batch = collate_combat(obs)
        has_rv = reqs[0][2] is not None
        rv = torch.from_numpy(np.concatenate([r[2] for r in reqs])) if has_rv else None
        if graphed is not None:
            al, tl = graphed(batch, rv)
        else:
            al, tl = policy_dists(policy, to_device(batch, dev), rv.to(dev) if rv is not None else None)
        al, tl = al.float().cpu().numpy(), tl.float().cpu().numpy()
        needs = batch["needs_target"].numpy()
        emask = batch["enemy_mask"].numpy()
        i = 0
        for wid, o, _ in reqs:
            n = len(o)
            resp_qs[wid].put((al[i:i + n], tl[i:i + n], needs[i:i + n], emask[i:i + n]))
            i += n
        stats["calls"] += 1
        stats["rows"] += rows
        stats["busy"] += time.perf_counter() - t0


class RemoteActor:
    """워커 쪽: 관측을 서버에 보내고 logit 을 받아 여기서 샘플링한다. actor 인터페이스는 LocalActor 와 같다."""

    def __init__(self, wid: int, req_q, resp_q):
        self.wid, self.req_q, self.resp_q = wid, req_q, resp_q

    def __call__(self, obs, run_vec, greedy: bool = False):
        rv = run_vec.detach().cpu().numpy().astype(np.float32) if run_vec is not None else None
        self.req_q.put((self.wid, obs, rv))
        al, tl, needs, emask = self.resp_q.get()
        return sample_from(torch.from_numpy(al), torch.from_numpy(tl), torch.from_numpy(needs),
                           torch.from_numpy(emask), greedy)


class InferenceServer:
    """학습 루프 쪽에서 서버 프로세스를 띄우고 가중치를 보낸다. 워커에게는 handles(i) 를 넘긴다."""

    def __init__(self, vocab, d_model: int, device: str, n_workers: int, max_batch: int = 512,
                 max_wait: float = 0.002, policy_kwargs: dict | None = None):
        ctx = mp.get_context("spawn")
        self.req_q = ctx.Queue()
        self.resp_qs = [ctx.Queue() for _ in range(n_workers)]
        self.id_q = ctx.Queue()  # 워커가 자기 번호를 하나씩 가져간다
        for i in range(n_workers):
            self.id_q.put(i)
        self._ctrl, child = ctx.Pipe()
        self.proc = ctx.Process(target=_serve, daemon=True,
                                args=(vocab, d_model, device, self.req_q, self.resp_qs, child, max_batch, max_wait,
                                      policy_kwargs))
        self.proc.start()

    def worker_args(self):
        return self.req_q, self.resp_qs, self.id_q

    def load(self, blob: bytes) -> None:
        self._ctrl.send(("weights", blob))
        if self._ctrl.recv()[0] != "ok":
            raise RuntimeError("추론 서버가 가중치를 읽지 못했다")

    def stats(self) -> dict:
        """지난 호출 뒤로 서버가 한 계산: 호출 수, 행 수(관측 수), GPU 계산에 쓴 초."""
        self._ctrl.send(("stats",))
        return self._ctrl.recv()[1]

    def close(self) -> None:
        if self.proc.is_alive():
            self._ctrl.send(("stop",))
            self.proc.join(timeout=10)
            if self.proc.is_alive():
                self.proc.terminate()


def attach_remote(agent, worker_args) -> None:
    """워커 초기화에서 부른다: 이 워커의 전투 정책 호출을 서버로 돌린다."""
    req_q, resp_qs, id_q = worker_args
    wid = id_q.get()
    agent.remote_actor = RemoteActor(wid, req_q, resp_qs[wid])
