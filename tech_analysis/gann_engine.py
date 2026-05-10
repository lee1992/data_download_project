import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tech_analysis.config import SystemConfig


# =============================================================================
# Module 2: 江恩理论引擎
# =============================================================================

@dataclass
class GannLevel:
    """一个江恩价格/时间位"""
    value: float  # 价格值或K线序号
    ratio: float  # 对应的江恩比例
    level_type: str  # 'price' 或 'time'
    source: str  # 来源描述 (如 "主低点-主高点 50%回撤")
    strength: float = 1.0  # 强度 (多个来源叠加时增加)


@dataclass
class GannGrid:
    """江恩网格输出"""
    price_levels: List[GannLevel]  # 关键价格位
    time_windows: List[GannLevel]  # 关键时间窗口
    resonance_points: List[dict]  # 价格+时间共振点
    resonance_matrix: Optional[np.ndarray] = None  # 共振矩阵 (可视化用)


class GannEngine:
    """江恩理论引擎: 输出价格+时间网格"""

    def __init__(self, config: SystemConfig):
        self.config = config

    def analyze(self, swings: pd.DataFrame,
                df: pd.DataFrame) -> GannGrid:
        """
        计算江恩价格网格和时间网格

        Parameters:
            swings: ZigZag摆动点
            df: 原始OHLCV数据 (用于确定当前bar位置)
        """
        # 1. 识别主要高低点 (取最高的高点和最低的低点, 以及次要的)
        major_swings = self._identify_major_swings(swings)

        # 2. 计算价格网格
        price_levels = self._calc_price_grid(major_swings)

        # 3. 计算时间网格
        time_windows = self._calc_time_grid(major_swings, len(df))

        # 4. 计算共振矩阵
        resonance_points = self._calc_resonance(price_levels, time_windows, df)

        return GannGrid(
            price_levels=price_levels,
            time_windows=time_windows,
            resonance_points=resonance_points
        )

    def _identify_major_swings(self, swings: pd.DataFrame) -> pd.DataFrame:
        """识别主要摆动点 (幅度最大的转折点)"""
        if len(swings) < 2:
            return swings

        # 计算每个摆动点前后的幅度
        prices = swings['price'].values
        importance = np.zeros(len(prices))

        for i in range(len(prices)):
            left_move = abs(prices[i] - prices[i - 1]) / prices[i - 1] if i > 0 else 0
            right_move = abs(prices[min(i + 1, len(prices) - 1)] - prices[i]) / prices[i] if i < len(prices) - 1 else 0
            importance[i] = max(left_move, right_move)

        swings = swings.copy()
        swings['importance'] = importance

        # 返回重要性排名前N的摆动点, 但至少保留所有
        return swings.sort_values('importance', ascending=False)

    def _calc_price_grid(self, swings: pd.DataFrame) -> List[GannLevel]:
        """计算江恩价格分割位"""
        levels = []

        if len(swings) < 2:
            return levels

        # 取所有高低点的两两组合
        highs = swings[swings['type'] == 'H']['price'].values
        lows = swings[swings['type'] == 'L']['price'].values

        if len(highs) == 0 or len(lows) == 0:
            return levels

        # 主要区间: 最高点-最低点
        overall_high = np.max(highs)
        overall_low = np.min(lows)

        pairs = [(overall_low, overall_high, "整体区间")]

        # 最近一段上涨/下跌
        if len(swings) >= 2:
            last_two = swings.head(2)  # 按importance排序的前两个
            p1, p2 = sorted(last_two['price'].values)
            pairs.append((p1, p2, "主要波段"))

        # 对每个价格对计算江恩分割
        seen = set()
        for low, high, source in pairs:
            span = high - low
            if span <= 0:
                continue
            for ratio in self.config.gann_price_ratios:
                price = low + span * ratio
                # 去重 (四舍五入到4位)
                key = round(price, 4)
                if key not in seen:
                    seen.add(key)
                    levels.append(GannLevel(
                        value=price,
                        ratio=ratio,
                        level_type='price',
                        source=f"{source} {ratio:.1%}",
                        strength=1.0
                    ))

        # 合并相近的价格位, 增加强度
        levels = self._merge_nearby_levels(levels, 'price')

        return sorted(levels, key=lambda x: x.value)

    def _calc_time_grid(self, swings: pd.DataFrame,
                        total_bars: int) -> List[GannLevel]:
        """计算江恩时间窗口"""
        levels = []

        if len(swings) < 2:
            return levels

        # 取主要摆动点之间的时间跨度
        bar_indices = swings['bar_index'].values

        # 主要时间跨度
        for i in range(len(bar_indices)):
            for j in range(i + 1, min(i + 4, len(bar_indices))):
                T = abs(bar_indices[j] - bar_indices[i])
                if T < 3:
                    continue
                start = max(bar_indices[i], bar_indices[j])

                for ratio in self.config.gann_time_ratios:
                    target_bar = int(start + T * ratio)
                    if 0 <= target_bar <= total_bars + total_bars * 0.5:
                        levels.append(GannLevel(
                            value=target_bar,
                            ratio=ratio,
                            level_type='time',
                            source=f"T={T}bars × {ratio:.3f}",
                            strength=1.0
                        ))

        # 合并相近的时间窗口
        levels = self._merge_nearby_levels(levels, 'time')

        return sorted(levels, key=lambda x: x.value)

    def _merge_nearby_levels(self, levels: List[GannLevel],
                             level_type: str) -> List[GannLevel]:
        """合并相近的价格位/时间窗口, 叠加强度"""
        if not levels:
            return levels

        if level_type == 'price':
            tolerance = self.config.gann_snap_tolerance / 100.0
        else:
            tolerance = self.config.gann_time_snap_bars

        sorted_levels = sorted(levels, key=lambda x: x.value)
        merged = [sorted_levels[0]]

        for lvl in sorted_levels[1:]:
            prev = merged[-1]
            if level_type == 'price':
                is_close = abs(lvl.value - prev.value) / max(prev.value, 1e-10) < tolerance
            else:
                is_close = abs(lvl.value - prev.value) <= tolerance

            if is_close:
                # 合并: 取平均价格, 叠加强度
                prev.value = (prev.value * prev.strength + lvl.value) / (prev.strength + 1)
                prev.strength += 1.0
                prev.source += f" | {lvl.source}"
            else:
                merged.append(lvl)

        return merged

    def _calc_resonance(self, price_levels: List[GannLevel],
                        time_windows: List[GannLevel],
                        df: pd.DataFrame) -> List[dict]:
        """计算价格-时间共振点"""
        resonance = []

        for tl in time_windows:
            bar_idx = int(tl.value)
            if bar_idx >= len(df):
                continue

            actual_high = df.iloc[bar_idx]['high'] if bar_idx < len(df) else None
            actual_low = df.iloc[bar_idx]['low'] if bar_idx < len(df) else None

            if actual_high is None:
                continue

            for pl in price_levels:
                tol = self.config.gann_snap_tolerance / 100.0
                if (actual_low <= pl.value <= actual_high or
                        abs(pl.value - actual_high) / actual_high < tol or
                        abs(pl.value - actual_low) / actual_low < tol):
                    score = (pl.strength * self.config.gann_resonance_weights['price_hit'] +
                             tl.strength * self.config.gann_resonance_weights['time_hit'] +
                             self.config.gann_resonance_weights['price_time_cross'])

                    resonance.append({
                        'bar_index': bar_idx,
                        'datetime': df.index[bar_idx] if bar_idx < len(df) else None,
                        'price_level': pl.value,
                        'time_ratio': tl.ratio,
                        'score': score,
                        'price_strength': pl.strength,
                        'time_strength': tl.strength,
                    })

        return sorted(resonance, key=lambda x: x['score'], reverse=True)

    def check_snap(self, price: float, price_levels: List[GannLevel]) -> Tuple[bool, Optional[GannLevel]]:
        """检查某个价格是否吸附在江恩位上"""
        tol = self.config.gann_snap_tolerance / 100.0
        for lvl in price_levels:
            if abs(price - lvl.value) / max(price, 1e-10) < tol:
                return True, lvl
        return False, None

