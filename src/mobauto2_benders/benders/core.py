from __future__ import annotations

from typing import Dict, List, Tuple


class CorePoint:
    """
    Maintains a running core point xbar for Benders (Magnanti–Wong).

    Tracks aggregated master starts per time bucket for OUT/RET directions:
      - Yout_bar[t] = moving average of sum_q yOUT[q,t]
      - Yret_bar[t] = moving average of sum_q yRET[q,t]

    Update rule with mixing factor alpha in (0, 1]:
        xbar <- (1 - alpha) * xbar + alpha * xk
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = float(alpha) if alpha is not None else 0.3
        if not (0.0 < self.alpha <= 1.0):
            self.alpha = 0.3
        self._yout_bar: List[float] | None = None
        self._yret_bar: List[float] | None = None

    def _ensure_len(self, n: int) -> None:
        if self._yout_bar is None or self._yret_bar is None:
            self._yout_bar = [0.0 for _ in range(n)]
            self._yret_bar = [0.0 for _ in range(n)]
            return
        if len(self._yout_bar) < n:
            self._yout_bar += [0.0 for _ in range(n - len(self._yout_bar))]
        if len(self._yret_bar) < n:
            self._yret_bar += [0.0 for _ in range(n - len(self._yret_bar))]

    def update_from_candidate(self, candidate: Dict[str, float], T_hint: int | None = None) -> None:
        # Determine T from candidate keys if not provided
        if T_hint is None:
            tmax = -1
            for k in candidate.keys():
                if not isinstance(k, str):
                    continue
                if k.startswith("yOUT[") or k.startswith("yRET["):
                    try:
                        inside = k[k.find("[") + 1 : k.find("]")]
                        _, t_str = inside.split(",")
                        tmax = max(tmax, int(t_str.strip()))
                    except Exception:
                        continue
            T = tmax + 1 if tmax >= 0 else 0
        else:
            T = int(T_hint)
        self._ensure_len(T)
        # Current aggregated starts per time
        cur_out = [0.0 for _ in range(T)]
        cur_ret = [0.0 for _ in range(T)]
        for name, val in candidate.items():
            if not isinstance(name, str):
                continue
            try:
                if name.startswith("yOUT["):
                    inside = name[name.find("[") + 1 : name.find("]")]
                    _, tau_str = inside.split(",")
                    tau = int(tau_str.strip())
                    if 0 <= tau < T:
                        cur_out[tau] += float(val)
                elif name.startswith("yRET["):
                    inside = name[name.find("[") + 1 : name.find("]")]
                    _, tau_str = inside.split(",")
                    tau = int(tau_str.strip())
                    if 0 <= tau < T:
                        cur_ret[tau] += float(val)
            except Exception:
                continue
        # Initialize if first time
        if self._yout_bar is None or self._yret_bar is None:
            self._yout_bar = list(cur_out)
            self._yret_bar = list(cur_ret)
            return
        # EMA update
        a = self.alpha
        for t in range(T):
            self._yout_bar[t] = (1.0 - a) * float(self._yout_bar[t]) + a * float(cur_out[t])
            self._yret_bar[t] = (1.0 - a) * float(self._yret_bar[t]) + a * float(cur_ret[t])

    def get(self) -> Tuple[list[float], list[float]]:
        return (list(self._yout_bar or []), list(self._yret_bar or []))

    def as_params(self) -> dict:
        yout, yret = self.get()
        return {"Yout": yout, "Yret": yret}


__all__ = ["CorePoint"]

