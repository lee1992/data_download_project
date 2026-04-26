"""
Elliott Wave Identification System — Phase 0 / 1 / 2
=====================================================
File   : ew_model.py
Author : EW-System
Date   : 2026-04-25
Desc   : Self-contained module for Elliott Wave identification
         from 5-min OHLCV CSV files.

Usage
-----
    from ew_model import Pipeline
    pipe = Pipeline("US_QQQ_K_5M_2026-04-22_2026-04-24.csv")
    pipe.run()                       # run full Phase 0→2
    pipe.print_report()              # text summary
    pipe.plot()                      # matplotlib chart

Dependencies
------------
    pandas, numpy, matplotlib  (all standard)
"""

from __future__ import annotations
import warnings, math, itertools
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Phase 0 — Data Engineering
# ---------------------------------------------------------------------------

class DataLoader:
    """Load, validate and optionally resample a 5-min OHLCV CSV."""

    # 标准列 rename mapping（你的 CSV → 内部列名）
    _COL_MAP = {
        "time_key": "datetime",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "code": "code",
    }

    # US 市场 RTH session（东部时间）
    _US_SESSION = ("09:30", "16:00")
    # A 股 session
    _CN_SESSION_AM = ("09:30", "11:30")
    _CN_SESSION_PM = ("13:00", "15:00")

    def __init__(self, path: str | Path, tz: str | None = None):
        self.path = Path(path)
        self.raw: pd.DataFrame = pd.DataFrame()
        self.df: pd.DataFrame = pd.DataFrame()
        self.issues: List[str] = []
        self._tz = tz  # if None, auto-detect from filename

    # ---- public API ----

    def load(self) -> pd.DataFrame:
        """Read CSV → standardise → validate → return clean df."""
        self._read_csv()
        self._standardise()
        self._infer_timezone()
        self._validate()
        self._mark_sessions()
        return self.df

    def resample(self, rule: str = "1h") -> pd.DataFrame:
        """
        Resample loaded 5-min data to a coarser frequency.
        Supports: '15min', '30min', '1h', '4h', 'D', 'W', 'M'

        IMPORTANT: resampling is done *per session-day* for intraday
        freqs to avoid blending overnight gaps.
        """
        if self.df.empty:
            raise RuntimeError("Call .load() first")
        return self._resample_impl(rule)

    # ---- internals ----

    def _read_csv(self):
        # handle BOM
        print(self.path)
        self.raw = pd.read_csv(self.path, encoding="utf-8-sig")

    def _standardise(self):
        df = self.raw.copy()
        # rename columns we care about
        rename = {k: v for k, v in self._COL_MAP.items() if k in df.columns}
        df.rename(columns=rename, inplace=True)

        # parse datetime
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # keep only what we need
        keep = ["datetime", "open", "high", "low", "close", "volume"]
        if "code" in df.columns:
            keep.insert(0, "code")
        df = df[[c for c in keep if c in df.columns]].copy()

        # numeric coerce
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        self.df = df

    def _infer_timezone(self):
        if self._tz:
            return
        code = ""
        if "code" in self.df.columns:
            code = str(self.df["code"].iloc[0])
        if code.startswith("US.") or code.startswith("NASDAQ") or code.startswith("NYSE"):
            self._tz = "US/Eastern"
        else:
            self._tz = "Asia/Shanghai"

    def _validate(self) -> List[str]:
        df = self.df
        issues = []

        # 1) duplicate timestamps
        dups = df["datetime"].duplicated()
        if dups.any():
            issues.append(f"Duplicate timestamps: {dups.sum()} rows")
            df = df[~dups].copy()

        # 2) OHLC sanity
        bad_hl = df["high"] < df["low"]
        if bad_hl.any():
            issues.append(f"high < low on {bad_hl.sum()} bars")

        bad_oh = df["high"] < df[["open", "close"]].max(axis=1)
        if bad_oh.any():
            n = bad_oh.sum()
            issues.append(f"high < max(open,close) on {n} bars (minor)")

        bad_ol = df["low"] > df[["open", "close"]].min(axis=1)
        if bad_ol.any():
            n = bad_ol.sum()
            issues.append(f"low > min(open,close) on {n} bars (minor)")

        # 3) nulls
        nulls = df[["open", "high", "low", "close"]].isna().any(axis=1)
        if nulls.any():
            issues.append(f"NaN in OHLC: {nulls.sum()} bars")
            df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        self.issues = issues
        self.df = df.reset_index(drop=True)
        return issues

    def _mark_sessions(self):
        """Add helper columns: trade_date, bar_of_day, is_gap."""
        df = self.df
        df["trade_date"] = df["datetime"].dt.date
        # bar-of-day sequential index (resets each day)
        df["bar_of_day"] = df.groupby("trade_date").cumcount()

        # mark overnight gaps
        df["is_gap"] = False
        dates = df["trade_date"].unique()
        for i in range(1, len(dates)):
            prev_mask = df["trade_date"] == dates[i - 1]
            curr_mask = df["trade_date"] == dates[i]
            if prev_mask.any() and curr_mask.any():
                prev_close = df.loc[prev_mask, "close"].iloc[-1]
                curr_open = df.loc[curr_mask, "open"].iloc[0]
                gap_pct = (curr_open - prev_close) / prev_close * 100
                first_idx = df.index[curr_mask][0]
                df.loc[first_idx, "is_gap"] = True
                df.loc[first_idx, "gap_pct"] = gap_pct

        self.df = df

    def _resample_impl(self, rule: str) -> pd.DataFrame:
        df = self.df.set_index("datetime")

        if rule in ("D", "W", "M"):
            # daily and above: group by trade_date then aggregate
            if rule == "D":
                grp = df.groupby("trade_date")
            elif rule == "W":
                df["_week"] = pd.to_datetime(df["trade_date"]).dt.to_period("W")
                grp = df.groupby("_week")
            else:
                df["_month"] = pd.to_datetime(df["trade_date"]).dt.to_period("M")
                grp = df.groupby("_month")

            agg = grp.agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
            agg.index.name = "datetime"
            return agg.reset_index()

        # intraday: resample per day to avoid blending gaps
        frames = []
        for date, gdf in df.groupby("trade_date"):
            r = gdf[["open", "high", "low", "close", "volume"]].resample(rule).agg(
                {"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"}
            ).dropna(subset=["open"])
            frames.append(r)
        out = pd.concat(frames)
        out.index.name = "datetime"
        return out.reset_index()


# ---------------------------------------------------------------------------
# Phase 1 — Pivot / ZigZag identification
# ---------------------------------------------------------------------------

@dataclass
class Pivot:
    """A single pivot (swing high or swing low)."""
    idx: int                 # positional index into the source DataFrame
    datetime: Any            # timestamp
    price: float             # the high (if swing high) or low (if swing low)
    kind: str                # 'H' or 'L'
    atr_at_point: float = 0.0

    def __repr__(self):
        return (f"Pivot({self.kind} idx={self.idx} "
                f"dt={self.datetime} px={self.price:.4f})")


@dataclass
class Segment:
    """A swing between two consecutive pivots."""
    start: Pivot
    end: Pivot
    direction: int           # +1 up, -1 down
    price_move: float        # absolute price change
    pct_move: float          # percentage change
    duration_bars: int
    duration_time: Any       # timedelta

    @property
    def length(self):
        return abs(self.price_move)


class PivotEngine:
    """
    Identify multi-scale pivot points using adaptive ZigZag.

    Parameters
    ----------
    atr_period : int
        ATR lookback (default 14 bars = 70 min on 5-min data)
    atr_mult : float
        Multiplier on ATR for the ZigZag threshold.
        Larger → fewer pivots → captures bigger waves.
    price_source : str
        'hl' (use high for swing highs, low for swing lows — recommended)
        'close' (use close for everything)
    hard_pivot_times : list[str|datetime], optional
        Timestamps that MUST appear as pivots (Phase-1 anchor).
        The engine will "snap" to the nearest bar within hard_pivot_window.
    hard_pivot_window : int
        Number of bars around each hard_pivot_time to search (default 2).
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_mult: float = 2.0,
        price_source: str = "hl",
        hard_pivot_times: list | None = None,
        hard_pivot_window: int = 2,
    ):
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.price_source = price_source
        self.hard_pivot_times = hard_pivot_times or []
        self.hard_pivot_window = hard_pivot_window

    # ---- public ----

    def find_pivots(self, df: pd.DataFrame) -> Tuple[List[Pivot], pd.Series]:
        """
        Returns (pivots, atr_series).
        df must have columns: datetime, open, high, low, close.
        """
        atr = self._compute_atr(df)
        pivots = self._zigzag(df, atr)

        # inject hard pivots
        if self.hard_pivot_times:
            pivots = self._inject_hard_pivots(df, atr, pivots)

        return pivots, atr

    def pivots_to_segments(self, pivots: List[Pivot]) -> List[Segment]:
        """Convert consecutive pivots to Segment objects."""
        segs = []
        for i in range(len(pivots) - 1):
            a, b = pivots[i], pivots[i + 1]
            d = 1 if b.price > a.price else -1
            move = b.price - a.price
            pct = move / a.price * 100 if a.price else 0
            dur_bars = b.idx - a.idx
            dur_time = b.datetime - a.datetime if isinstance(b.datetime, pd.Timestamp) else None
            segs.append(Segment(a, b, d, move, pct, dur_bars, dur_time))
        return segs

    # ---- internals ----

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        n = len(df)
        tr = np.empty(n)
        tr[0] = h[0] - l[0]
        for i in range(1, n):
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        atr = pd.Series(tr, index=df.index).rolling(self.atr_period, min_periods=1).mean()
        return atr

    def _zigzag(self, df: pd.DataFrame, atr: pd.Series) -> List[Pivot]:
        """Core adaptive ZigZag algorithm."""
        n = len(df)
        if n < 2:
            return []

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        dts = df["datetime"].values
        atr_vals = atr.values

        use_hl = (self.price_source == "hl")

        # --- init ---
        pivots: List[Pivot] = []
        direction = 0  # 0=undecided, 1=looking_for_high, -1=looking_for_low

        last_high_idx = 0
        last_high_px = highs[0] if use_hl else closes[0]
        last_low_idx = 0
        last_low_px = lows[0] if use_hl else closes[0]

        for i in range(1, n):
            hi = highs[i] if use_hl else closes[i]
            lo = lows[i] if use_hl else closes[i]
            thr = atr_vals[i] * self.atr_mult

            if direction == 0:
                # undecided: first significant move sets direction
                if hi - last_low_px >= thr:
                    # significant up move → last_low was a pivot low
                    pivots.append(Pivot(last_low_idx, dts[last_low_idx],
                                       last_low_px, "L", atr_vals[last_low_idx]))
                    direction = 1
                    last_high_idx = i
                    last_high_px = hi
                elif last_high_px - lo >= thr:
                    pivots.append(Pivot(last_high_idx, dts[last_high_idx],
                                       last_high_px, "H", atr_vals[last_high_idx]))
                    direction = -1
                    last_low_idx = i
                    last_low_px = lo
                else:
                    # track running extremes
                    if hi > last_high_px:
                        last_high_idx, last_high_px = i, hi
                    if lo < last_low_px:
                        last_low_idx, last_low_px = i, lo

            elif direction == 1:
                # trending up, looking for higher high or reversal
                if hi > last_high_px:
                    last_high_idx, last_high_px = i, hi
                elif last_high_px - lo >= thr:
                    # reversal down → confirm last high as pivot
                    pivots.append(Pivot(last_high_idx, dts[last_high_idx],
                                       last_high_px, "H", atr_vals[last_high_idx]))
                    direction = -1
                    last_low_idx = i
                    last_low_px = lo

            else:  # direction == -1
                if lo < last_low_px:
                    last_low_idx, last_low_px = i, lo
                elif hi - last_low_px >= thr:
                    pivots.append(Pivot(last_low_idx, dts[last_low_idx],
                                       last_low_px, "L", atr_vals[last_low_idx]))
                    direction = 1
                    last_high_idx = i
                    last_high_px = hi

        # close out last pending pivot
        if direction == 1:
            pivots.append(Pivot(last_high_idx, dts[last_high_idx],
                               last_high_px, "H", atr_vals[last_high_idx]))
        elif direction == -1:
            pivots.append(Pivot(last_low_idx, dts[last_low_idx],
                               last_low_px, "L", atr_vals[last_low_idx]))

        # ensure alternation H-L-H-L...
        pivots = self._enforce_alternation(pivots)

        return pivots

    @staticmethod
    def _enforce_alternation(pivots: List[Pivot]) -> List[Pivot]:
        """Remove consecutive same-kind pivots, keeping the more extreme."""
        if len(pivots) < 2:
            return pivots
        clean = [pivots[0]]
        for p in pivots[1:]:
            if p.kind == clean[-1].kind:
                # same kind → keep the more extreme
                if p.kind == "H":
                    if p.price >= clean[-1].price:
                        clean[-1] = p
                else:
                    if p.price <= clean[-1].price:
                        clean[-1] = p
            else:
                clean.append(p)
        return clean

    def _inject_hard_pivots(
        self, df: pd.DataFrame, atr: pd.Series, pivots: List[Pivot]
    ) -> List[Pivot]:
        """Ensure hard_pivot_times appear in the pivot list."""
        existing_idxs = {p.idx for p in pivots}
        dts = df["datetime"].values

        for ts in self.hard_pivot_times:
            ts = pd.Timestamp(ts)
            # find nearest bar
            diffs = np.abs(pd.to_datetime(dts) - ts)
            candidates = np.argsort(diffs)[:self.hard_pivot_window + 1]
            best = None
            for c in candidates:
                if c not in existing_idxs:
                    best = c
                    break
            if best is None:
                continue  # already exists
            # decide kind: if local high ≈ price, 'H'; else 'L'
            hi = df["high"].iloc[best]
            lo = df["low"].iloc[best]
            cl = df["close"].iloc[best]
            # crude heuristic
            kind = "H" if cl >= (hi + lo) / 2 else "L"
            px = hi if kind == "H" else lo
            new_p = Pivot(int(best), dts[best], px, kind, atr.iloc[best])
            pivots.append(new_p)

        # re-sort and re-enforce alternation
        pivots.sort(key=lambda p: p.idx)
        pivots = self._enforce_alternation(pivots)
        return pivots


# ---------------------------------------------------------------------------
# Phase 2 — Impulse Wave Identification
# ---------------------------------------------------------------------------

# Fibonacci constants
_FIB = {
    "0.236": 0.236, "0.382": 0.382, "0.5": 0.500,
    "0.618": 0.618, "0.786": 0.786,
    "1.0": 1.000, "1.272": 1.272, "1.618": 1.618,
    "2.0": 2.000, "2.618": 2.618,
}
_FIB_RETRACE = [0.236, 0.382, 0.5, 0.618, 0.786]
_FIB_EXTEND  = [1.0, 1.272, 1.618, 2.0, 2.618]


@dataclass
class ImpulseCount:
    """One candidate impulse (5-wave motive) labeling."""
    pivot_indices: List[int]   # 6 pivot indices: [w0, w1, w2, w3, w4, w5]
    pivots: List[Pivot]        # the 6 Pivot objects
    direction: str             # 'up' or 'down'
    score: float = 0.0        # composite guideline score (higher = better)
    details: Dict[str, Any] = field(default_factory=dict)

    # convenience properties
    @property
    def w0(self): return self.pivots[0]
    @property
    def w1(self): return self.pivots[1]
    @property
    def w2(self): return self.pivots[2]
    @property
    def w3(self): return self.pivots[3]
    @property
    def w4(self): return self.pivots[4]
    @property
    def w5(self): return self.pivots[5]

    @property
    def invalidation(self) -> float:
        """Price level that invalidates this count."""
        if self.direction == "up":
            return self.w0.price   # wave 2 cannot go below wave 0
        else:
            return self.w0.price

    def summary(self) -> str:
        d = "↑" if self.direction == "up" else "↓"
        lines = [
            f"Impulse {d}  score={self.score:.3f}  invalidation={self.invalidation:.4f}",
            f"  W0 (start) : {self.w0.datetime}  {self.w0.price:.4f}",
            f"  W1 (end)   : {self.w1.datetime}  {self.w1.price:.4f}",
            f"  W2 (end)   : {self.w2.datetime}  {self.w2.price:.4f}",
            f"  W3 (end)   : {self.w3.datetime}  {self.w3.price:.4f}",
            f"  W4 (end)   : {self.w4.datetime}  {self.w4.price:.4f}",
            f"  W5 (end)   : {self.w5.datetime}  {self.w5.price:.4f}",
        ]
        for k, v in self.details.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class ImpulseSearcher:
    """
    Search for valid 5-wave impulse structures among a set of pivots.

    Parameters
    ----------
    pivots : list[Pivot]
        Alternating H/L pivots from PivotEngine.
    direction : str
        'up' for bullish impulse, 'down' for bearish.
    anchors : dict, optional
        Structural anchors (Phase-2 level).
        Keys can be: 'w0','w1','w2','w3','w4','w5'
        Values: pivot index (int) or datetime that will be snapped
        to the nearest pivot.
    top_k : int
        Number of best candidates to return.
    """

    def __init__(
        self,
        pivots: List[Pivot],
        direction: str = "up",
        anchors: Dict[str, Any] | None = None,
        top_k: int = 3,
    ):
        self.pivots = pivots
        self.direction = direction
        self.anchors = anchors or {}
        self.top_k = top_k

    # ---- public ----

    def search(self) -> List[ImpulseCount]:
        """Run search and return top_k scored candidates."""
        raw = self._enumerate_candidates()
        scored = []
        for combo in raw:
            ic = self._build_count(combo)
            if ic is not None:
                scored.append(ic)
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[: self.top_k]

    # ---- candidate enumeration ----

    def _enumerate_candidates(self) -> List[Tuple[int, ...]]:
        """
        Enumerate valid 6-pivot combos [w0,w1,w2,w3,w4,w5].

        For an UP impulse the pattern is: L H L H L H
        For a DOWN impulse:               H L H L H L

        We use constrained combinatorial search with pruning.
        """
        pvts = self.pivots
        n = len(pvts)

        if self.direction == "up":
            pattern = ["L", "H", "L", "H", "L", "H"]
        else:
            pattern = ["H", "L", "H", "L", "H", "L"]

        # build per-slot candidate lists
        slot_candidates: List[List[int]] = [[] for _ in range(6)]
        for slot in range(6):
            anchor_key = f"w{slot}"
            if anchor_key in self.anchors:
                # resolve anchor to pivot index
                a = self.anchors[anchor_key]
                idx = self._resolve_anchor(a)
                if idx is not None:
                    slot_candidates[slot] = [idx]
                continue
            for pi in range(n):
                if pvts[pi].kind == pattern[slot]:
                    slot_candidates[slot].append(pi)

        # if any slot has 0 candidates, impossible
        for slot in range(6):
            if len(slot_candidates[slot]) == 0:
                return []

        # enumerate combos with ordering constraint: w0 < w1 < ... < w5
        results = []
        self._dfs(slot_candidates, 0, [], results, n)
        return results

    def _dfs(self, slots, depth, current, results, n, max_results=50000):
        if len(results) >= max_results:
            return
        if depth == 6:
            results.append(tuple(current))
            return
        lo = current[-1] + 1 if current else 0
        for pi in slots[depth]:
            if pi < lo:
                continue
            # pruning: must leave room for remaining slots
            remaining = 5 - depth
            if pi + remaining >= n and remaining > 0:
                # need at least 'remaining' more pivots after pi
                if pi + remaining > n - 1:
                    continue
            current.append(pi)
            self._dfs(slots, depth + 1, current, results, n, max_results)
            current.pop()

    # ---- build & score ----

    def _build_count(self, combo: Tuple[int, ...]) -> Optional[ImpulseCount]:
        """Check hard rules; if pass, score guidelines."""
        pvts = [self.pivots[i] for i in combo]
        up = (self.direction == "up")

        # extract prices
        p0, p1, p2, p3, p4, p5 = [p.price for p in pvts]

        # wave lengths (signed, but we use absolute for comparison)
        len1 = p1 - p0  # wave 1
        len2 = p2 - p1  # wave 2 (retracement, opposite sign to len1)
        len3 = p3 - p2  # wave 3
        len4 = p4 - p3  # wave 4
        len5 = p5 - p4  # wave 5

        if up:
            a1, a3, a5 = len1, len3, len5    # motive legs (positive)
            r2, r4 = -len2, -len4              # corrections (positive)
        else:
            a1, a3, a5 = -len1, -len3, -len5
            r2, r4 = len2, len4

        # ==== HARD RULES (violate → discard) ====

        # Rule 0: motive waves must be in correct direction
        if a1 <= 0 or a3 <= 0 or a5 <= 0:
            return None
        if r2 <= 0 or r4 <= 0:
            return None

        # Rule 1: wave 2 cannot retrace more than 100% of wave 1
        if r2 >= a1:
            return None

        # Rule 2: wave 3 must NOT be the shortest of 1, 3, 5
        if a3 < a1 and a3 < a5:
            return None

        # Rule 3: wave 4 must not enter wave 1 territory
        if up:
            if p4 <= p1:
                return None
        else:
            if p4 >= p1:
                return None

        # ==== GUIDELINES (score) ====
        details = {}
        scores = []

        # G1: wave 2 retrace ratio (ideal: 0.382–0.618 of wave 1)
        retr2 = r2 / a1 if a1 else 0
        details["retr2"] = round(retr2, 4)
        s_r2 = self._fib_closeness(retr2, _FIB_RETRACE)
        scores.append(("retr2_fib", s_r2, 1.0))

        # G2: wave 3 extension ratio (ideal: 1.618 of wave 1)
        ext3 = a3 / a1 if a1 else 0
        details["ext3"] = round(ext3, 4)
        s_e3 = self._fib_closeness(ext3, _FIB_EXTEND)
        scores.append(("ext3_fib", s_e3, 1.5))  # higher weight

        # G3: wave 3 is longest (strong guideline)
        if a3 >= a1 and a3 >= a5:
            s_longest = 1.0
        elif a3 >= a1 or a3 >= a5:
            s_longest = 0.6
        else:
            s_longest = 0.2
        scores.append(("w3_longest", s_longest, 1.2))

        # G4: wave 4 retrace ratio (ideal: 0.236–0.382 of wave 3)
        retr4 = r4 / a3 if a3 else 0
        details["retr4"] = round(retr4, 4)
        s_r4 = self._fib_closeness(retr4, [0.236, 0.382, 0.5])
        scores.append(("retr4_fib", s_r4, 0.8))

        # G5: wave 5 / wave 1 ratio (ideal: 0.618, 1.0, 1.618)
        rat5 = a5 / a1 if a1 else 0
        details["rat5_over_w1"] = round(rat5, 4)
        s_r5 = self._fib_closeness(rat5, [0.618, 1.0, 1.618])
        scores.append(("w5_w1_ratio", s_r5, 1.0))

        # G6: alternation — wave 2 and wave 4 should differ in depth
        # heuristic: if their retrace ratios differ by > 0.15, good
        alt_diff = abs(retr2 - retr4)
        s_alt = min(1.0, alt_diff / 0.3)
        scores.append(("alternation", s_alt, 0.7))

        # G7: time proportions — wave 3 duration >= wave 1 duration
        t1 = pvts[1].idx - pvts[0].idx
        t3 = pvts[3].idx - pvts[2].idx
        t5 = pvts[5].idx - pvts[4].idx
        details["t1_bars"] = t1
        details["t3_bars"] = t3
        details["t5_bars"] = t5
        s_time = 0.5
        if t3 >= t1:
            s_time += 0.25
        if t1 > 0 and 0.3 <= t5 / t1 <= 3.0:
            s_time += 0.25
        scores.append(("time_prop", s_time, 0.6))

        # G8: overall trend alignment — total wave covers decent % move
        total_move_pct = abs(p5 - p0) / p0 * 100 if p0 else 0
        details["total_move_pct"] = round(total_move_pct, 2)
        s_size = min(1.0, total_move_pct / 2.0)   # at least 2% is nice
        scores.append(("size", s_size, 0.5))

        # composite
        total_w = sum(w for _, _, w in scores)
        composite = sum(s * w for _, s, w in scores) / total_w if total_w else 0
        details["score_breakdown"] = {name: round(s, 3) for name, s, _ in scores}

        return ImpulseCount(
            pivot_indices=list(combo),
            pivots=pvts,
            direction=self.direction,
            score=round(composite, 4),
            details=details,
        )

    @staticmethod
    def _fib_closeness(value: float, targets: List[float]) -> float:
        """Score how close `value` is to any Fibonacci target (0–1)."""
        if not targets:
            return 0.5
        min_dist = min(abs(value - t) for t in targets)
        # sigmoid-like decay: distance 0 → score 1, distance 0.5 → score ~0.22
        return math.exp(-3.0 * min_dist)

    def _resolve_anchor(self, anchor) -> Optional[int]:
        """Resolve an anchor value to a pivot list index."""
        if isinstance(anchor, int):
            if 0 <= anchor < len(self.pivots):
                return anchor
            return None
        # treat as datetime
        ts = pd.Timestamp(anchor)
        best_dist = None
        best_idx = None
        for i, p in enumerate(self.pivots):
            d = abs(pd.Timestamp(p.datetime) - ts)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx


# ---------------------------------------------------------------------------
# Pipeline — glue it all together
# ---------------------------------------------------------------------------

class Pipeline:
    """
    End-to-end Phase 0→2 pipeline.

    Usage
    -----
        pipe = Pipeline("path/to/file.csv")
        pipe.run()
        pipe.print_report()
        pipe.plot()
    """

    def __init__(
        self,
        csv_path: str | Path,
        # Phase 1 params
        atr_period: int = 14,
        atr_mult: float = 2.0,
        price_source: str = "hl",
        hard_pivot_times: list | None = None,
        hard_pivot_window: int = 2,
        # Phase 2 params
        direction: str = "auto",   # 'up', 'down', or 'auto' (detect)
        anchors: dict | None = None,
        top_k: int = 3,
    ):
        self.csv_path = csv_path
        self._p1_kwargs = dict(
            atr_period=atr_period, atr_mult=atr_mult,
            price_source=price_source,
            hard_pivot_times=hard_pivot_times,
            hard_pivot_window=hard_pivot_window,
        )
        self._direction = direction
        self._anchors = anchors or {}
        self._top_k = top_k

        # results
        self.df: pd.DataFrame = pd.DataFrame()
        self.pivots: List[Pivot] = []
        self.atr: pd.Series = pd.Series(dtype=float)
        self.segments: List[Segment] = []
        self.candidates: List[ImpulseCount] = []
        self.issues: List[str] = []

    def run(self):
        """Execute Phase 0 → 1 → 2."""
        # Phase 0
        loader = DataLoader(self.csv_path)
        self.df = loader.load()
        self.issues = loader.issues

        # Phase 1
        engine = PivotEngine(**self._p1_kwargs)
        self.pivots, self.atr = engine.find_pivots(self.df)
        self.segments = engine.pivots_to_segments(self.pivots)

        # auto-detect direction
        direction = self._direction
        if direction == "auto":
            if len(self.pivots) >= 2:
                first_px = self.pivots[0].price
                last_px = self.pivots[-1].price
                direction = "up" if last_px > first_px else "down"
            else:
                direction = "up"

        # Phase 2 — search both directions if auto
        searcher_up = ImpulseSearcher(
            self.pivots, direction="up",
            anchors=self._anchors, top_k=self._top_k
        )
        searcher_dn = ImpulseSearcher(
            self.pivots, direction="down",
            anchors=self._anchors, top_k=self._top_k
        )

        cands_up = searcher_up.search()
        cands_dn = searcher_dn.search()

        if direction == "up":
            self.candidates = cands_up if cands_up else cands_dn
        elif direction == "down":
            self.candidates = cands_dn if cands_dn else cands_up
        else:
            # merge and sort
            all_c = cands_up + cands_dn
            all_c.sort(key=lambda c: c.score, reverse=True)
            self.candidates = all_c[: self._top_k]

    def print_report(self):
        """Print a text summary of results."""
        print("=" * 70)
        print("ELLIOTT WAVE PIPELINE — REPORT")
        print("=" * 70)

        # Phase 0
        print(f"\n--- Phase 0: Data ---")
        print(f"  File          : {self.csv_path}")
        print(f"  Bars          : {len(self.df)}")
        if not self.df.empty:
            print(f"  Date range    : {self.df['datetime'].iloc[0]} → "
                  f"{self.df['datetime'].iloc[-1]}")
            print(f"  Price range   : {self.df['low'].min():.4f} → "
                  f"{self.df['high'].max():.4f}")
        if self.issues:
            print(f"  Issues        : {self.issues}")
        else:
            print(f"  Issues        : None ✓")

        # Phase 1
        print(f"\n--- Phase 1: Pivots ---")
        print(f"  Pivot count   : {len(self.pivots)}")
        print(f"  Segments      : {len(self.segments)}")
        if self.segments:
            avg_move = np.mean([abs(s.pct_move) for s in self.segments])
            print(f"  Avg seg move  : {avg_move:.2f}%")
        print(f"  Pivots detail :")
        for p in self.pivots:
            print(f"    {p}")

        # Phase 2
        print(f"\n--- Phase 2: Impulse Candidates (top {self._top_k}) ---")
        if not self.candidates:
            print("  No valid impulse found with current parameters.")
            print("  Try adjusting atr_mult (smaller → more pivots → more combos)")
        for i, c in enumerate(self.candidates):
            print(f"\n  ── Candidate #{i + 1} ──")
            print("  " + c.summary().replace("\n", "\n  "))

        print("\n" + "=" * 70)

    def plot(self):
        """Plot price, pivots, and best impulse candidate."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            print("matplotlib not installed, skipping plot.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(18, 10),
                                 gridspec_kw={"height_ratios": [3, 1]},
                                 sharex=True)
        ax_price = axes[0]
        ax_vol = axes[1]

        df = self.df
        x = df["datetime"]

        # candlestick-like: just use close line + high/low band
        ax_price.fill_between(x, df["low"], df["high"],
                              alpha=0.15, color="steelblue", label="H-L range")
        ax_price.plot(x, df["close"], linewidth=0.8, color="black", label="Close")

        # pivots
        for p in self.pivots:
            color = "red" if p.kind == "H" else "green"
            marker = "v" if p.kind == "H" else "^"
            ax_price.plot(p.datetime, p.price, marker=marker,
                          color=color, markersize=10, zorder=5)
            ax_price.annotate(f"{p.price:.2f}",
                              (p.datetime, p.price),
                              textcoords="offset points",
                              xytext=(0, 12 if p.kind == "L" else -16),
                              fontsize=7, ha="center", color=color)

        # zigzag line
        if self.pivots:
            zx = [p.datetime for p in self.pivots]
            zy = [p.price for p in self.pivots]
            ax_price.plot(zx, zy, "b--", linewidth=1.0, alpha=0.6, label="ZigZag")

        # best impulse
        if self.candidates:
            best = self.candidates[0]
            labels = ["W0", "W1", "W2", "W3", "W4", "W5"]
            colors_imp = ["gray", "#1f77b4", "#ff7f0e",
                          "#2ca02c", "#d62728", "#9467bd"]
            for j in range(6):
                p = best.pivots[j]
                ax_price.annotate(
                    labels[j],
                    (p.datetime, p.price),
                    textcoords="offset points",
                    xytext=(8, 8 if p.kind == "L" else -14),
                    fontsize=11, fontweight="bold",
                    color=colors_imp[j],
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="yellow", alpha=0.7),
                    zorder=10,
                )
            # draw impulse path
            ix = [best.pivots[j].datetime for j in range(6)]
            iy = [best.pivots[j].price for j in range(6)]
            ax_price.plot(ix, iy, "m-", linewidth=2.0, alpha=0.8,
                          label=f"Best impulse (score={best.score:.3f})")

        ax_price.set_title("Phase 0-2: Price + Pivots + Best Impulse", fontsize=14)
        ax_price.legend(loc="upper left", fontsize=8)
        ax_price.grid(True, alpha=0.3)

        # volume
        ax_vol.bar(x, df["volume"], width=0.003, color="steelblue", alpha=0.6)
        ax_vol.set_ylabel("Volume")
        ax_vol.grid(True, alpha=0.3)

        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig("ew_phase02_output.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Chart saved to ew_phase02_output.png")


# ---------------------------------------------------------------------------
# Convenience: run from command line
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    #path = sys.argv[1] if len(sys.argv) > 1 else "'.\\futu_data\\K_5M\\US_QQQ_K_5M_2010-01-01_2027-04-21.csv'"
    path = Path("futu_data") / "K_5M" / "US_QQQ_K_5M_2010-01-01_2027-04-21.csv"

    pipe = Pipeline(
        path,
        atr_period=14,
        atr_mult=1.5,       # slightly smaller → more pivots for 3 days of 5min data
        price_source="hl",
        direction="auto",
        top_k=3,
    )
    pipe.run()
    pipe.print_report()

    try:
        pipe.plot()
    except Exception as e:
        print(f"Plotting failed (headless?): {e}")