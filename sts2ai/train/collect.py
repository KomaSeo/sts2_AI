"""런 플레이(평가기 스냅숏 + GRPO 분기 전투)를 여러 프로세스로 나눠 돌린다.

학습 시간의 대부분은 배치 1 짜리 정책 호출을 반복하는 Python 루프다. 이런 계산은 GPU 나
PyTorch 스레드를 늘려도 빨라지지 않고, 런 하나를 프로세스 하나가 맡게 나누는 쪽이 코어 수만큼 빨라진다.

- 워커마다 환경(make_run())과 CPU 에이전트 복사본을 하나씩 가진다. PyTorch 스레드는 1 개.
- 반복마다 학습된 가중치를 바이트로 직렬화해 보내고, 워커는 버전이 바뀌었을 때만 다시 읽는다.
- 런 k 의 난수(numpy 와 torch 둘 다)는 seed_k 하나로 정해진다. 그래서 워커 수(0 = 현재 프로세스)와 상관없이 같은 시드면
  같은 결과가 나온다.
- Windows 는 spawn 방식이라 make_run 은 pickle 할 수 있어야 한다 (모듈 최상위 클래스나
  functools.partial). 스크립트는 `if __name__ == "__main__":` 아래에서 실행해야 한다.
"""

from __future__ import annotations

import io
import multiprocessing as mp
import os
from typing import Callable

import numpy as np
import torch

from ..envs.base import RunSim
from .loop import Agent, Config, play_run

_W: dict = {}  # 워커 프로세스 전역 상태


def default_workers(runs_per_iter: int) -> int:
    """논리 코어의 절반(하이퍼스레딩 제외 근사)과 반복당 런 수 중 작은 값."""
    return max(1, min(runs_per_iter, (os.cpu_count() or 2) // 2))


def _serialize(agent: Agent) -> bytes:
    buf = io.BytesIO()
    torch.save({k: {n: t.detach().cpu() for n, t in sd.items()} for k, sd in agent.state_dict().items()}, buf)
    return buf.getvalue()


def _init_worker(make_run: Callable[[], RunSim], cfg: Config, server_args=None) -> None:
    torch.set_num_threads(1)
    run = make_run()
    agent = Agent(run.vocab, cfg.d_model)
    if server_args is not None:
        from .infer_server import attach_remote  # noqa: PLC0415

        attach_remote(agent, server_args)
    _W.update(run=run, cfg=cfg, agent=agent, version=-1)


def _play(run: RunSim, agent: Agent, cfg: Config, seed: int, use_evaluator: bool, collect_grpo: bool, greedy: bool,
          start=None, probe_weights=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)  # 정책 샘플링은 torch 전역 난수를 쓴다. 런마다 고정해야 프로세스와 무관해진다
    return play_run(agent, run, cfg, rng, seed, use_evaluator=use_evaluator, collect_grpo=collect_grpo, greedy=greedy,
                    start=start, probe_weights=probe_weights)


def _work(task):
    version, blob, seed, use_evaluator, collect_grpo, greedy, start, probe_weights = task
    agent: Agent = _W["agent"]
    if _W["version"] != version:
        agent.load_state_dict(torch.load(io.BytesIO(blob), map_location="cpu"))
        _W["version"] = version
    snaps, groups, res = _play(_W["run"], agent, _W["cfg"], seed, use_evaluator, collect_grpo, greedy, start,
                               probe_weights)
    # 텐서는 numpy 로 바꿔 보낸다. torch 텐서를 그대로 보내면 공유 메모리 핸들로 넘어가서
    # 작은 텐서가 많을 때 Windows 에서 핸들이 쌓인다.
    for g in groups:
        for t in g:
            if isinstance(t.run_vec, torch.Tensor):
                t.run_vec = t.run_vec.numpy()
    return snaps, groups, res


class RunCollector:
    """런 여러 판을 플레이해서 (스냅숏, GRPO 그룹, 결과) 를 시드 순서대로 돌려준다.

    workers == 0 이면 현재 프로세스에서 run 으로 순서대로 돌린다.
    """

    def __init__(self, cfg: Config, workers: int, run: RunSim | None = None,
                 make_run: Callable[[], RunSim] | None = None, vocab=None, server_device: str = "cuda"):
        """cfg.infer_server 면(워커가 있을 때만) 워커들의 전투 정책 호출을 server_device 의 추론 서버 하나가 모아 계산한다."""
        self.cfg = cfg
        self.workers = workers
        self.run = run
        self.pool = None
        self.server = None
        self._version = 0
        self._blob: bytes | None = None
        if workers > 0:
            if make_run is None:
                raise ValueError("workers > 0 이면 make_run (pickle 가능한 환경 생성 함수) 이 필요하다")
            server_args = None
            if getattr(cfg, "infer_server", False):
                from .infer_server import InferenceServer  # noqa: PLC0415

                self.server = InferenceServer(vocab if vocab is not None else make_run().vocab, cfg.d_model,
                                              server_device, workers)
                server_args = self.server.worker_args()
            self.pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker,
                                                     initargs=(make_run, cfg, server_args))
        elif run is None:
            self.run = make_run()

    def sync(self, agent: Agent) -> None:
        """학습이 끝난 가중치를 다음 수집에 쓰도록 등록한다 (추론 서버에는 바로 보낸다)."""
        if self.pool is not None:
            self._version += 1
            self._blob = _serialize(agent)
            if self.server is not None:
                self.server.load(self._blob)

    def collect(self, actor: Agent, seeds: list[int], use_evaluator: bool, collect_grpo: bool = True,
                greedy: bool = False, starts: list | None = None, probe_weights: dict | None = None) -> list[tuple]:
        """starts[i] 가 있으면 i 번째 런은 그 막 시작 상태에서 시작한다. probe_weights: 모의전 적 고르기 가중치."""
        starts = starts or [None] * len(seeds)
        if self.pool is None:
            return [_play(self.run, actor, self.cfg, s, use_evaluator, collect_grpo, greedy, st, probe_weights)
                    for s, st in zip(seeds, starts)]
        if self._blob is None:
            self.sync(actor)
        tasks = [(self._version, self._blob, s, use_evaluator, collect_grpo, greedy, st, probe_weights)
                 for s, st in zip(seeds, starts)]
        return self.pool.map(_work, tasks, chunksize=1)

    def server_stats(self) -> dict:
        return self.server.stats() if self.server is not None else {}

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
        if self.server is not None:
            self.server.close()
            self.server = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
