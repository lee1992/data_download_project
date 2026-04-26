"""
Elliott Wave Identification System — Phase 0 ~ 2
=================================================
Phase 0: 数据工程 (data.py)
Phase 1: 极值点识别 (pivots.py)
Phase 1.5: 锚点系统 (anchors.py)
Phase 2: 浪形搜索与评分 (wave_search.py, scoring.py)

数据格式要求：CSV，至少包含以下列：
  code, time_key, open, close, high, low, volume

Author: Claude  |  License: MIT
"""

# ============================================================
#                    PHASE 0: 数据工程
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union
from enum import Enum, auto
import warnings
import copy
import matplotlib
matplotlib.use('Agg')  # 关键：Agg 是纯文件渲染后端，不依赖 GUI
# ────────────────────────────────────────────────────────────
#  0.1  加载与标准化
# ────────────────────────────────────────────────────────────

def load_ohlcv(path: str,
               time_col: str = 'time_key',
               code_col: str = 'code',
               encoding: str = 'utf-8-sig') -> pd.DataFrame:
    """
    读取 CSV 并标准化为内部格式。

    返回 DataFrame 列名统一为：
        code, datetime, open, high, low, close, volume
    索引为自增整数，按 datetime 升序排列。
    """
    df = pd.read_csv(path, encoding=encoding)

    # ── 列名映射 ──
    rename_map = {
        time_col: 'datetime',
        'open':   'open',
        'high':   'high',
        'low':    'low',
        'close':  'close',
        'volume': 'volume',
    }
    if code_col in df.columns:
        rename_map[code_col] = 'code'

    df = df.rename(columns=rename_map)

    # 只保留需要的列
    keep = [c for c in ['code', 'datetime', 'open', 'high', 'low', 'close', 'volume']
            if c in df.columns]
    df = df[keep].copy()

    # ── 类型转换 ──
    df['datetime'] = pd.to_datetime(df['datetime'])
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)

    # ── 排序 & 去重 ──
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.drop_duplicates(subset=['datetime'], keep='first').reset_index(drop=True)

    return df


# ────────────────────────────────────────────────────────────
#  0.2  数据质量校验
# ────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    total_bars: int = 0
    duplicates_removed: int = 0
    nan_rows: int = 0
    ohlc_violations: int = 0          # high < max(open,close) 等
    time_gaps: List[Tuple] = field(default_factory=list)   # (idx, gap_minutes)
    is_valid: bool = True
    messages: List[str] = field(default_factory=list)


def validate_ohlcv(df: pd.DataFrame,
                   expected_interval_min: int = 5,
                   max_gap_mult: float = 3.0) -> ValidationReport:
    """
    检查 OHLCV 数据质量，返回 ValidationReport。
    """
    rpt = ValidationReport()
    rpt.total_bars = len(df)

    # NaN
    nan_mask = df[['open', 'high', 'low', 'close']].isna().any(axis=1)
    rpt.nan_rows = int(nan_mask.sum())
    if rpt.nan_rows > 0:
        rpt.messages.append(f"⚠ {rpt.nan_rows} rows contain NaN in OHLC")

    # OHLC 逻辑
    mask = (
        (df['high'] < df['open']) | (df['high'] < df['close']) |
        (df['low'] > df['open']) | (df['low'] > df['close']) |
        (df['high'] < df['low'])
    )
    rpt.ohlc_violations = int(mask.sum())
    if rpt.ohlc_violations > 0:
        rpt.messages.append(f"⚠ {rpt.ohlc_violations} bars violate OHLC logic")

    # 时间间隔缺口（只在同一交易日内检测）
    if len(df) > 1:
        dt = df['datetime']
        diffs = dt.diff().dt.total_seconds() / 60  # minutes
        threshold = expected_interval_min * max_gap_mult
        for i in range(1, len(diffs)):
            if pd.notna(diffs.iloc[i]):
                gap = diffs.iloc[i]
                # 跨日缺口不报（>12小时视为跨日/周末）
                if gap > threshold and gap < 720:
                    rpt.time_gaps.append((i, gap))
        if rpt.time_gaps:
            rpt.messages.append(
                f"⚠ {len(rpt.time_gaps)} intra-session gaps > {threshold} min"
            )

    rpt.is_valid = (rpt.nan_rows == 0 and rpt.ohlc_violations == 0)
    if rpt.is_valid:
        rpt.messages.append("✔ Data passed all checks")
    return rpt


# ────────────────────────────────────────────────────────────
#  0.3  重采样：5min → 15min / 30min / 1H / 4H / D / W / M
# ────────────────────────────────────────────────────────────

RESAMPLE_MAP = {
    '15min': '15min',  '15T': '15min',
    '30min': '30min',  '30T': '30min',
    '1H':    '60min',   '60min': '60min',
    '4H':    '240min',   '240min': '240min',
    'D':     '1D',   '1D': '1D',
    'W':     '1W',   '1W': '1W',
    'M':     '1M',   '1M': '1M',
    "Q":"QE" # 月末
}


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    将 OHLCV DataFrame 重采样到更高频率。
    df 必须含有 'datetime' 列（已 pd.to_datetime）。

    target_tf: '15min','30min','1H','4H','D','W','M'
    """
    rule = RESAMPLE_MAP.get(target_tf)
    if rule is None:
        raise ValueError(f"Unknown timeframe: {target_tf}. "
                         f"Choose from {list(RESAMPLE_MAP.keys())}")

    tmp = df.set_index('datetime')
    agg = tmp.resample(rule).agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum',
    }).dropna(subset=['open'])

    result = agg.reset_index()

    # 保留 code 列（如果存在）
    if 'code' in df.columns:
        result['code'] = df['code'].iloc[0]

    return result


def build_multi_tf(df_5min: pd.DataFrame,
                   timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    从 5min 基础数据构建多频率字典。
    返回 {'5min': df, '15min': df, '1H': df, '4H': df, 'D': df, ...}
    """
    if timeframes is None:
        timeframes = ['15min', '30min', '1H', '4H', 'D']

    result = {'5min': df_5min.copy()}
    for tf in timeframes:
        result[tf] = resample_ohlcv(df_5min, tf)
    return result


# ============================================================
#                  PHASE 1: 极值点识别
# ============================================================

# ────────────────────────────────────────────────────────────
#  1.0  基础类型定义
# ────────────────────────────────────────────────────────────

class PivotType(Enum):
    HIGH = auto()
    LOW  = auto()


@dataclass
class Pivot:
    """一个极值点（拐点）"""
    idx: int                    # 在 DataFrame 中的行索引
    datetime: datetime          # 时间戳
    price: float                # 极值价格（high 或 low）
    pivot_type: PivotType       # HIGH 或 LOW
    bar_high: float = 0.0       # 该 bar 的 high
    bar_low: float = 0.0        # 该 bar 的 low
    bar_close: float = 0.0      # 该 bar 的 close
    scale: float = 0.0          # 所属尺度（ATR 倍数）
    is_anchor: bool = False     # 是否为用户指定的锚点
    anchor_info: Optional[Dict] = None  # 锚点附加信息

    def __repr__(self):
        t = 'H' if self.pivot_type == PivotType.HIGH else 'L'
        anc = ' [ANCHOR]' if self.is_anchor else ''
        return f"Pivot({t} {self.price:.2f} @ {self.datetime}{anc})"


@dataclass
class Segment:
    """两个相邻 pivot 之间的一段走势"""
    start: Pivot
    end: Pivot
    direction: int              # +1 上涨, -1 下跌
    price_change: float         # 绝对价格变动
    pct_change: float           # 百分比变动
    duration_bars: int          # 持续 bar 数
    duration_minutes: float     # 持续分钟数

    def __repr__(self):
        d = '↑' if self.direction == 1 else '↓'
        return (f"Seg({d} {self.pct_change:+.2f}% "
                f"{self.duration_bars}bars "
                f"{self.start.price:.2f}→{self.end.price:.2f})")


# ────────────────────────────────────────────────────────────
#  1.1  ATR 计算
# ────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    计算 Average True Range。
    返回与 df 等长的 Series（前 period-1 个值用扩展窗口）。
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    tr = np.zeros(len(df))
    tr[0] = high[0] - low[0]
    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )

    atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().values
    return pd.Series(atr, index=df.index)


# ────────────────────────────────────────────────────────────
#  1.2  自适应 ZigZag
# ────────────────────────────────────────────────────────────

def zigzag_atr(df: pd.DataFrame,
               atr_period: int = 14,
               atr_mult: float = 2.0,
               use_hl: bool = True) -> List[Pivot]:
    """
    自适应 ZigZag 极值点识别。

    参数：
        df:          标准化的 OHLCV DataFrame
        atr_period:  ATR 计算周期
        atr_mult:    ATR 倍数作为 ZigZag 阈值
        use_hl:      True=用 high/low 做极值, False=用 close

    返回：Pivot 列表（高低交替出现）
    """
    atr = compute_atr(df, atr_period).values
    highs = df['high'].values if use_hl else df['close'].values
    lows = df['low'].values if use_hl else df['close'].values
    closes = df['close'].values
    datetimes = df['datetime'].values

    n = len(df)
    if n < 2:
        return []

    pivots: List[Pivot] = []
    direction = 0        # 0=未定, 1=找高点中, -1=找低点中
    last_high_idx = 0
    last_low_idx = 0
    last_high_val = highs[0]
    last_low_val = lows[0]

    for i in range(1, n):
        threshold = atr[i] * atr_mult
        if threshold < 1e-10:
            threshold = abs(closes[i]) * 0.001  # fallback

        if direction == 0:
            # 初始化：判断第一步方向
            if highs[i] - last_low_val >= threshold:
                # 确认了一个低点 → 方向转为寻找高点
                pivots.append(Pivot(
                    idx=last_low_idx,
                    datetime=pd.Timestamp(datetimes[last_low_idx]),
                    price=last_low_val,
                    pivot_type=PivotType.LOW,
                    bar_high=highs[last_low_idx],
                    bar_low=lows[last_low_idx],
                    bar_close=closes[last_low_idx],
                ))
                direction = 1
                last_high_idx = i
                last_high_val = highs[i]

            elif last_high_val - lows[i] >= threshold:
                # 确认了一个高点 → 方向转为寻找低点
                pivots.append(Pivot(
                    idx=last_high_idx,
                    datetime=pd.Timestamp(datetimes[last_high_idx]),
                    price=last_high_val,
                    pivot_type=PivotType.HIGH,
                    bar_high=highs[last_high_idx],
                    bar_low=lows[last_high_idx],
                    bar_close=closes[last_high_idx],
                ))
                direction = -1
                last_low_idx = i
                last_low_val = lows[i]

            else:
                # 更新候选
                if highs[i] > last_high_val:
                    last_high_idx = i
                    last_high_val = highs[i]
                if lows[i] < last_low_val:
                    last_low_idx = i
                    last_low_val = lows[i]

        elif direction == 1:
            # 正在寻找高点（上升趋势中）
            if highs[i] > last_high_val:
                last_high_idx = i
                last_high_val = highs[i]

            elif last_high_val - lows[i] >= threshold:
                # 从高点回落超过阈值 → 确认高点
                pivots.append(Pivot(
                    idx=last_high_idx,
                    datetime=pd.Timestamp(datetimes[last_high_idx]),
                    price=last_high_val,
                    pivot_type=PivotType.HIGH,
                    bar_high=highs[last_high_idx],
                    bar_low=lows[last_high_idx],
                    bar_close=closes[last_high_idx],
                ))
                direction = -1
                last_low_idx = i
                last_low_val = lows[i]

        elif direction == -1:
            # 正在寻找低点（下降趋势中）
            if lows[i] < last_low_val:
                last_low_idx = i
                last_low_val = lows[i]

            elif highs[i] - last_low_val >= threshold:
                # 从低点反弹超过阈值 → 确认低点
                pivots.append(Pivot(
                    idx=last_low_idx,
                    datetime=pd.Timestamp(datetimes[last_low_idx]),
                    price=last_low_val,
                    pivot_type=PivotType.LOW,
                    bar_high=highs[last_low_idx],
                    bar_low=lows[last_low_idx],
                    bar_close=closes[last_low_idx],
                ))
                direction = 1
                last_high_idx = i
                last_high_val = highs[i]

    # 添加最后一个未确认的极值点
    if direction == 1:
        pivots.append(Pivot(
            idx=last_high_idx,
            datetime=pd.Timestamp(datetimes[last_high_idx]),
            price=last_high_val,
            pivot_type=PivotType.HIGH,
            bar_high=highs[last_high_idx],
            bar_low=lows[last_high_idx],
            bar_close=closes[last_high_idx],
        ))
    elif direction == -1:
        pivots.append(Pivot(
            idx=last_low_idx,
            datetime=pd.Timestamp(datetimes[last_low_idx]),
            price=last_low_val,
            pivot_type=PivotType.LOW,
            bar_high=highs[last_low_idx],
            bar_low=lows[last_low_idx],
            bar_close=closes[last_low_idx],
        ))

    return pivots


# ────────────────────────────────────────────────────────────
#  1.3  多尺度极值点
# ────────────────────────────────────────────────────────────

class MultiScalePivots:
    """
    用不同 ATR 倍数生成多层级极值点。

    大尺度极值点 ⊂ 小尺度极值点（在 price/time 容差内）。
    这为后续嵌套浪形分析提供自然的层级基础。
    """

    def __init__(self,
                 df: pd.DataFrame,
                 scales: List[float] = None,
                 atr_period: int = 14,
                 merge_tolerance_bars: int = 3):
        """
        参数：
            df:     标准化 OHLCV DataFrame
            scales: ATR 倍数列表，从小到大
            atr_period: ATR 周期
            merge_tolerance_bars: 合并同一 pivot 的 bar 容差
        """
        if scales is None:
            scales = [1.0, 2.0, 3.5, 6.0, 10.0]

        self.df = df
        self.scales = sorted(scales)
        self.atr_period = atr_period
        self.merge_tolerance = merge_tolerance_bars

        # 为每个尺度生成 pivots
        self.pivot_sets: Dict[float, List[Pivot]] = {}
        for s in self.scales:
            pvts = zigzag_atr(df, atr_period=atr_period, atr_mult=s)
            # 标记尺度
            for p in pvts:
                p.scale = s
            self.pivot_sets[s] = pvts

    def get_pivots(self, scale: float) -> List[Pivot]:
        """获取指定尺度的 pivot 列表"""
        return self.pivot_sets.get(scale, [])

    def verify_nesting(self, tolerance_bars: int = None) -> List[str]:
        """
        验证大尺度极值点是否（在容差内）是小尺度极值点的子集。
        返回问题列表（空 = 全部通过）。
        """
        tol = tolerance_bars or self.merge_tolerance
        issues = []

        for i in range(len(self.scales) - 1):
            small_scale = self.scales[i]
            large_scale = self.scales[i + 1]
            small_pivots = self.pivot_sets[small_scale]
            large_pivots = self.pivot_sets[large_scale]

            small_indices = {p.idx for p in small_pivots}

            for lp in large_pivots:
                # 检查大尺度 pivot 是否在小尺度 pivot 集合附近
                found = False
                for offset in range(-tol, tol + 1):
                    if (lp.idx + offset) in small_indices:
                        found = True
                        break
                if not found:
                    issues.append(
                        f"Scale {large_scale} pivot at idx={lp.idx} "
                        f"({lp.datetime}) not found in scale {small_scale}"
                    )

        return issues

    def summary(self) -> str:
        """打印各尺度 pivot 数量摘要"""
        lines = ["MultiScalePivots Summary:"]
        for s in self.scales:
            n = len(self.pivot_sets[s])
            lines.append(f"  Scale {s:>5.1f}x ATR : {n:>4d} pivots")
        return '\n'.join(lines)


# ────────────────────────────────────────────────────────────
#  1.4  Pivot → Segment 转换
# ────────────────────────────────────────────────────────────

def pivots_to_segments(pivots: List[Pivot]) -> List[Segment]:
    """
    将 pivot 列表转换为 segment（波段）列表。
    每个 segment 代表两个相邻 pivot 之间的一段走势。
    """
    segments = []
    for i in range(len(pivots) - 1):
        p1 = pivots[i]
        p2 = pivots[i + 1]

        price_change = p2.price - p1.price
        direction = 1 if price_change > 0 else -1
        pct = (price_change / p1.price) * 100 if p1.price != 0 else 0
        dur_bars = p2.idx - p1.idx
        dur_min = (p2.datetime - p1.datetime).total_seconds() / 60

        segments.append(Segment(
            start=p1,
            end=p2,
            direction=direction,
            price_change=abs(price_change),
            pct_change=pct,
            duration_bars=dur_bars,
            duration_minutes=dur_min,
        ))

    return segments


# ============================================================
#              PHASE 1.5: 锚点系统 (Anchor System)
# ============================================================

class AnchorStrength(Enum):
    """锚点强度等级"""
    SOFT = 'soft'               # 软提示：提高评分权重
    HARD_PIVOT = 'hard_pivot'   # 硬锚点：必须是 pivot
    STRUCTURAL = 'structural'   # 结构锚点：必须是某浪的端点


@dataclass
class Anchor:
    """
    用户指定的先验锚点。

    用法示例：
        # "2025-04-07 的低点大概率是一个重要拐点"
        Anchor(datetime=..., strength=AnchorStrength.HARD_PIVOT,
               pivot_type=PivotType.LOW)

        # "这个点必须是某个 impulse 的 wave-3 终点"
        Anchor(datetime=..., strength=AnchorStrength.STRUCTURAL,
               pivot_type=PivotType.HIGH,
               structural_role='wave3_end')

        # "这附近大概是一个高点，但不确定"
        Anchor(datetime=..., strength=AnchorStrength.SOFT,
               pivot_type=PivotType.HIGH)
    """
    datetime: datetime                              # 时间点
    strength: AnchorStrength                        # 锚点强度
    pivot_type: Optional[PivotType] = None          # HIGH/LOW（可选）
    structural_role: Optional[str] = None           # 结构角色
    price: Optional[float] = None                   # 价格（可选，用于匹配验证）
    time_tolerance_bars: int = 5                    # 时间匹配容差（bar 数）
    note: str = ''                                  # 备注

    # structural_role 可选值：
    #   'wave1_start', 'wave1_end',
    #   'wave2_end',
    #   'wave3_end',
    #   'wave4_end',
    #   'wave5_end',
    #   'waveA_end', 'waveB_end', 'waveC_end'
    #   或任意自定义字符串

    def __repr__(self):
        return (f"Anchor({self.strength.value} @ {self.datetime}, "
                f"role={self.structural_role}, type={self.pivot_type})")


class AnchorManager:
    """
    管理锚点，在 Phase 1 和 Phase 2 中应用约束。
    """

    def __init__(self):
        self.anchors: List[Anchor] = []

    def add(self, anchor: Anchor):
        self.anchors.append(anchor)

    def add_pivot_anchor(self,
                         dt: Union[str, datetime],
                         pivot_type: PivotType,
                         strength: AnchorStrength = AnchorStrength.HARD_PIVOT,
                         structural_role: str = None,
                         price: float = None,
                         tolerance_bars: int = 5,
                         note: str = ''):
        """便捷方法：添加一个锚点"""
        if isinstance(dt, str):
            dt = pd.Timestamp(dt)
        self.anchors.append(Anchor(
            datetime=dt,
            strength=strength,
            pivot_type=pivot_type,
            structural_role=structural_role,
            price=price,
            time_tolerance_bars=tolerance_bars,
            note=note,
        ))

    # ─── Phase 1 应用：确保 pivot 集合包含锚点 ───

    def apply_to_pivots(self,
                        pivots: List[Pivot],
                        df: pd.DataFrame) -> List[Pivot]:
        """
        Phase 1 层面的锚点应用。

        对于 HARD_PIVOT 和 STRUCTURAL 级别的锚点：
          - 如果 pivot 列表中已有匹配 → 标记为 anchor
          - 如果没有匹配 → 强制插入一个 pivot

        对于 SOFT 级别：
          - 如果有匹配 → 标记（后续评分加权用）
          - 没有则不强制

        返回：修改后的 pivot 列表（仍保持高低交替）
        """
        result = list(pivots)
        datetimes = df['datetime'].values

        for anchor in self.anchors:
            # 在 df 中找到最近的 bar
            target_idx = self._find_nearest_bar(anchor.datetime, datetimes)
            if target_idx is None:
                warnings.warn(f"Anchor at {anchor.datetime} outside data range")
                continue

            # 在现有 pivots 中查找匹配
            matched_pivot = self._find_matching_pivot(
                result, target_idx, anchor.time_tolerance_bars
            )

            if matched_pivot is not None:
                # 已有匹配 pivot → 标记
                matched_pivot.is_anchor = True
                matched_pivot.anchor_info = {
                    'strength': anchor.strength,
                    'structural_role': anchor.structural_role,
                    'note': anchor.note,
                }
            elif anchor.strength in (AnchorStrength.HARD_PIVOT,
                                     AnchorStrength.STRUCTURAL):
                # 没有匹配 → 强制插入
                new_pivot = self._create_pivot_from_anchor(anchor, target_idx, df)
                result = self._insert_pivot_sorted(result, new_pivot)
                # 插入后需要修复高低交替性
                result = self._fix_alternation(result)

        return result

    # ─── Phase 2 应用：返回结构约束 ───

    def get_structural_constraints(self) -> List[Dict]:
        """
        返回所有 STRUCTURAL 锚点的约束，供 Phase 2 浪形搜索使用。

        返回格式：
        [
            {
                'datetime': datetime,
                'pivot_idx': int or None,  # 匹配到的 pivot 在 list 中的位置
                'role': 'wave3_end',
                'pivot_type': PivotType.HIGH,
                'tolerance_bars': 5,
            },
            ...
        ]
        """
        constraints = []
        for a in self.anchors:
            if a.strength == AnchorStrength.STRUCTURAL and a.structural_role:
                constraints.append({
                    'datetime': a.datetime,
                    'pivot_idx': None,   # Phase 2 搜索时填充
                    'role': a.structural_role,
                    'pivot_type': a.pivot_type,
                    'tolerance_bars': a.time_tolerance_bars,
                })
        return constraints

    # ─── 内部方法 ───

    def _find_nearest_bar(self, dt, datetimes) -> Optional[int]:
        """在 datetime 数组中找最近的索引"""
        dt_np = np.datetime64(dt)
        diffs = np.abs(datetimes.astype('datetime64[ns]') - dt_np)
        idx = int(np.argmin(diffs))
        return idx

    def _find_matching_pivot(self,
                             pivots: List[Pivot],
                             target_idx: int,
                             tolerance: int) -> Optional[Pivot]:
        """在 pivot 列表中找到 idx 在 tolerance 范围内的 pivot"""
        for p in pivots:
            if abs(p.idx - target_idx) <= tolerance:
                return p
        return None

    def _create_pivot_from_anchor(self,
                                  anchor: Anchor,
                                  bar_idx: int,
                                  df: pd.DataFrame) -> Pivot:
        """从锚点信息创建一个新 Pivot"""
        row = df.iloc[bar_idx]
        pt = anchor.pivot_type or PivotType.HIGH  # 默认
        price = row['high'] if pt == PivotType.HIGH else row['low']
        if anchor.price is not None:
            price = anchor.price

        return Pivot(
            idx=bar_idx,
            datetime=pd.Timestamp(row['datetime']),
            price=price,
            pivot_type=pt,
            bar_high=row['high'],
            bar_low=row['low'],
            bar_close=row['close'],
            is_anchor=True,
            anchor_info={
                'strength': anchor.strength,
                'structural_role': anchor.structural_role,
                'note': anchor.note,
            }
        )

    def _insert_pivot_sorted(self,
                             pivots: List[Pivot],
                             new_pivot: Pivot) -> List[Pivot]:
        """按 idx 排序插入 pivot"""
        result = list(pivots)
        inserted = False
        for i, p in enumerate(result):
            if new_pivot.idx < p.idx:
                result.insert(i, new_pivot)
                inserted = True
                break
        if not inserted:
            result.append(new_pivot)
        return result

    def _fix_alternation(self, pivots: List[Pivot]) -> List[Pivot]:
        """
        修复 pivot 列表的高低交替性。
        如果出现连续两个 HIGH 或两个 LOW，保留极值更极端的那个。
        """
        if len(pivots) <= 1:
            return pivots

        fixed = [pivots[0]]
        for i in range(1, len(pivots)):
            if pivots[i].pivot_type == fixed[-1].pivot_type:
                # 同类冲突 → 保留更极端的
                if pivots[i].pivot_type == PivotType.HIGH:
                    if pivots[i].price > fixed[-1].price:
                        fixed[-1] = pivots[i]
                else:
                    if pivots[i].price < fixed[-1].price:
                        fixed[-1] = pivots[i]
            else:
                fixed.append(pivots[i])

        return fixed


# ============================================================
#       可视化辅助（用 matplotlib，可选依赖）
# ============================================================

def plot_pivots(df: pd.DataFrame,
                pivots: List[Pivot],
                title: str = 'ZigZag Pivots',
                figsize: Tuple = (16, 6)):
    """
    绘制 K 线 + ZigZag 折线 + pivot 标注。
    需要 matplotlib。
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib not installed, skip plotting")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # K 线简化（用 close 折线代替蜡烛图，方便）
    ax.plot(df['datetime'], df['close'], color='gray', alpha=0.5,
            linewidth=0.5, label='Close')

    # ZigZag 折线
    if pivots:
        zz_x = [p.datetime for p in pivots]
        zz_y = [p.price for p in pivots]
        ax.plot(zz_x, zz_y, 'b-', linewidth=1.2, alpha=0.8, label='ZigZag')

        # 标注 pivot 点
        for p in pivots:
            color = 'red' if p.pivot_type == PivotType.HIGH else 'green'
            marker = 'v' if p.pivot_type == PivotType.HIGH else '^'
            size = 100 if p.is_anchor else 40
            edge = 'gold' if p.is_anchor else color
            ax.scatter(p.datetime, p.price, c=color, marker=marker,
                       s=size, zorder=5, edgecolors=edge, linewidths=1.5)

    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_multi_scale(df: pd.DataFrame,
                     msp: MultiScalePivots,
                     scales_to_show: List[float] = None,
                     figsize: Tuple = (18, 10)):
    """
    绘制多尺度 pivot 对比图（多子图）。
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib not installed")
        return

    if scales_to_show is None:
        scales_to_show = msp.scales

    n = len(scales_to_show)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    colors = ['blue', 'orange', 'red', 'purple', 'darkred']

    for i, scale in enumerate(scales_to_show):
        ax = axes[i]
        pvts = msp.get_pivots(scale)

        ax.plot(df['datetime'], df['close'], color='gray',
                alpha=0.4, linewidth=0.5)

        if pvts:
            zz_x = [p.datetime for p in pvts]
            zz_y = [p.price for p in pvts]
            c = colors[i % len(colors)]
            ax.plot(zz_x, zz_y, color=c, linewidth=1.5, alpha=0.9)

            for p in pvts:
                mc = 'red' if p.pivot_type == PivotType.HIGH else 'green'
                mk = 'v' if p.pivot_type == PivotType.HIGH else '^'
                ax.scatter(p.datetime, p.price, c=mc, marker=mk,
                           s=30, zorder=5)

        ax.set_ylabel(f'Scale {scale}x', fontsize=10)
        ax.set_title(f'ATR×{scale} — {len(pvts)} pivots', fontsize=10)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()
    fig.suptitle('Multi-Scale Pivot Comparison', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# ============================================================
#  Phase 0~1 端到端演示（使用你的 QQQ 数据）
# ============================================================

def demo_phase_0_1():
    """
    用你的 QQQ 5min CSV 做端到端演示。
    """
    print("=" * 60)
    print("  PHASE 0: 数据加载与验证")
    print("=" * 60)

    # 加载
    df = load_ohlcv('.\\futu_data\\K_5M\\US_QQQ_K_5M_2010-01-01_2027-04-21.csv')
    print(f"\n✔ Loaded {len(df)} bars")
    print(f"  Code:  {df['code'].iloc[0] if 'code' in df.columns else 'N/A'}")
    print(f"  Range: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")
    print(f"  Price: {df['low'].min():.2f} ~ {df['high'].max():.2f}")

    # 验证
    rpt = validate_ohlcv(df, expected_interval_min=5)
    print(f"\n  Validation:")
    for msg in rpt.messages:
        print(f"    {msg}")

    # 多频率
    print(f"\n  Multi-timeframe:")
    mtf = build_multi_tf(df, ['15min', '30min', '1H'])
    for tf, d in mtf.items():
        print(f"    {tf:>6s}: {len(d):>4d} bars")

    print("\n" + "=" * 60)
    print("  PHASE 1: 极值点识别")
    print("=" * 60)

    # 单尺度 ZigZag
    pivots = zigzag_atr(df, atr_period=14, atr_mult=2.0)
    segments = pivots_to_segments(pivots)
    print(f"\n✔ ZigZag (ATR×2.0): {len(pivots)} pivots, {len(segments)} segments")
    print(f"  Pivots:")
    for p in pivots:
        print(f"    {p}")
    print(f"\n  Segments:")
    for s in segments:
        print(f"    {s}")

    # 多尺度
    msp = MultiScalePivots(df, scales=[1.0, 2.0, 3.5, 6.0])
    print(f"\n{msp.summary()}")

    nesting_issues = msp.verify_nesting(tolerance_bars=5)
    if nesting_issues:
        print(f"\n  ⚠ Nesting issues:")
        for issue in nesting_issues:
            print(f"    {issue}")
    else:
        print(f"\n  ✔ All scales pass nesting check")

    # 锚点演示
    print("\n" + "=" * 60)
    print("  PHASE 1.5: 锚点系统演示")
    print("=" * 60)

    anchor_mgr = AnchorManager()

    # 假设用户认为 4月23日 13:50 的低点是一个重要结构点
    anchor_mgr.add_pivot_anchor(
        dt='2026-04-23 13:50:00',
        pivot_type=PivotType.LOW,
        strength=AnchorStrength.STRUCTURAL,
        structural_role='wave4_end',
        note='User: 关税恐慌低点，可能是4浪终点'
    )

    # 应用锚点到 pivot 集合
    pivots_with_anchors = anchor_mgr.apply_to_pivots(pivots, df)
    print(f"\n✔ After applying anchors: {len(pivots_with_anchors)} pivots")
    for p in pivots_with_anchors:
        print(f"    {p}")

    # 获取结构约束（传给 Phase 2）
    constraints = anchor_mgr.get_structural_constraints()
    print(f"\n  Structural constraints for Phase 2:")
    for c in constraints:
        print(f"    {c}")

    # 可视化
    print("\n  (Plotting...)")
    try:
        plot_pivots(df, pivots_with_anchors,
                    title='QQQ 5min — ZigZag ATR×2.0 with Anchors')
    except Exception as e:
        print(f"  Plot skipped: {e}")

    return df, pivots_with_anchors, msp, anchor_mgr


# ============================================================
#  如果直接运行此文件
# ============================================================

if __name__ == '__main__':
    df, pivots, msp, anchor_mgr = demo_phase_0_1()