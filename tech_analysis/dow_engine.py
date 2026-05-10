import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tech_analysis.config import SystemConfig

# =============================================================================
# Module 1: 道氏理论引擎
# =============================================================================

class TrendDirection(Enum):
    UP = "上涨"
    DOWN = "下跌"
    NEUTRAL = "震荡"


@dataclass
class DowConstraint:
    """道氏理论输出的约束条件"""
    primary_trend: TrendDirection  # 主趋势方向
    secondary_trend: TrendDirection  # 次级趋势方向
    valid_highs: List[Tuple[str, float]]  # 有效高点序列 [(datetime, price), ...]
    valid_lows: List[Tuple[str, float]]  # 有效低点序列
    invalidation_price: float  # 趋势失效价格 (跌破此价=主趋势改变)
    allowed_wave_types: List[str]  # 允许的波浪结构类型
    forbidden_wave_types: List[str]  # 禁止的波浪结构类型
    confidence: float  # 判定置信度 0-1


class DowEngine:
    """道氏理论引擎: 输出方向约束"""

    def __init__(self, config: SystemConfig):
        self.config = config

    def analyze(self, swings: pd.DataFrame,
                current_price: float) -> DowConstraint:
        """
        分析摆动点序列, 输出道氏约束

        Parameters:
            swings: ZigZag输出的摆动点DataFrame
            current_price: 当前最新价格
        """
        if len(swings) < 4:
            return DowConstraint(
                primary_trend=TrendDirection.NEUTRAL,
                secondary_trend=TrendDirection.NEUTRAL,
                valid_highs=[], valid_lows=[],
                invalidation_price=0,
                allowed_wave_types=['impulse_up', 'impulse_down', 'corrective'],
                forbidden_wave_types=[],
                confidence=0.0
            )

        # 提取有效高点和低点序列
        highs = swings[swings['type'] == 'H'][['datetime', 'price']].values.tolist()
        lows = swings[swings['type'] == 'L'][['datetime', 'price']].values.tolist()

        # 过滤有效摆动 (回调幅度 > dow_valid_swing_pct)
        valid_highs, valid_lows = self._filter_valid_swings(swings)

        # 判定主趋势
        primary_trend, confidence = self._determine_trend(valid_highs, valid_lows)

        # 判定次级趋势 (用最近N个摆动点)
        recent_n = min(6, len(swings))
        recent_swings = swings.tail(recent_n)
        recent_highs = recent_swings[recent_swings['type'] == 'H'][['datetime', 'price']].values.tolist()
        recent_lows = recent_swings[recent_swings['type'] == 'L'][['datetime', 'price']].values.tolist()
        secondary_trend, _ = self._determine_trend(
            [(str(h[0]), h[1]) for h in recent_highs],
            [(str(l[0]), l[1]) for l in recent_lows]
        )

        # 确定失效价格
        invalidation_price = self._get_invalidation_price(
            primary_trend, valid_lows, valid_highs, current_price
        )

        # 确定允许/禁止的波浪类型
        allowed, forbidden = self._get_wave_constraints(primary_trend, secondary_trend)

        return DowConstraint(
            primary_trend=primary_trend,
            secondary_trend=secondary_trend,
            valid_highs=valid_highs,
            valid_lows=valid_lows,
            invalidation_price=invalidation_price,
            allowed_wave_types=allowed,
            forbidden_wave_types=forbidden,
            confidence=confidence
        )

    def _filter_valid_swings(self, swings: pd.DataFrame) -> Tuple[list, list]:
        """过滤出有效摆动点 (回调幅度超过阈值)"""
        threshold = self.config.dow_valid_swing_pct / 100.0
        valid_highs = []
        valid_lows = []

        prices = swings['price'].values
        types = swings['type'].values
        dts = swings['datetime'].values

        for i in range(1, len(swings)):
            move_pct = abs(prices[i] - prices[i - 1]) / prices[i - 1]
            if move_pct >= threshold:
                if types[i] == 'H':
                    valid_highs.append((str(dts[i]), float(prices[i])))
                else:
                    valid_lows.append((str(dts[i]), float(prices[i])))

        return valid_highs, valid_lows

    def _determine_trend(self, highs: list, lows: list) -> Tuple[TrendDirection, float]:
        """通过高低点序列判定趋势方向"""
        if len(highs) < 2 and len(lows) < 2:
            return TrendDirection.NEUTRAL, 0.0

        higher_highs = 0
        lower_highs = 0
        if len(highs) >= 2:
            for i in range(1, len(highs)):
                if highs[i][1] > highs[i - 1][1]:
                    higher_highs += 1
                else:
                    lower_highs += 1

        higher_lows = 0
        lower_lows = 0
        if len(lows) >= 2:
            for i in range(1, len(lows)):
                if lows[i][1] > lows[i - 1][1]:
                    higher_lows += 1
                else:
                    lower_lows += 1

        up_score = higher_highs + higher_lows
        down_score = lower_highs + lower_lows
        total = up_score + down_score

        if total == 0:
            return TrendDirection.NEUTRAL, 0.0

        if up_score > down_score:
            return TrendDirection.UP, up_score / total
        elif down_score > up_score:
            return TrendDirection.DOWN, down_score / total
        else:
            return TrendDirection.NEUTRAL, 0.5

    def _get_invalidation_price(self, trend: TrendDirection,
                                valid_lows: list, valid_highs: list,
                                current_price: float) -> float:
        """确定趋势失效价格"""
        margin = self.config.dow_breakout_margin / 100.0

        if trend == TrendDirection.UP and valid_lows:
            # 上涨趋势: 跌破最近有效低点 → 趋势失效
            last_valid_low = valid_lows[-1][1]
            return last_valid_low * (1 - margin)
        elif trend == TrendDirection.DOWN and valid_highs:
            # 下跌趋势: 突破最近有效高点 → 趋势失效
            last_valid_high = valid_highs[-1][1]
            return last_valid_high * (1 + margin)
        else:
            return current_price

    def _get_wave_constraints(self, primary: TrendDirection,
                              secondary: TrendDirection) -> Tuple[list, list]:
        """根据趋势判定输出波浪约束"""
        if primary == TrendDirection.UP:
            allowed = ['impulse_up', 'corrective_down', 'diagonal_up']
            forbidden = ['impulse_down']
            if secondary == TrendDirection.DOWN:
                # 次级回调中, 允许调整浪
                allowed.append('corrective_complex')
        elif primary == TrendDirection.DOWN:
            allowed = ['impulse_down', 'corrective_up', 'diagonal_down']
            forbidden = ['impulse_up']
            if secondary == TrendDirection.UP:
                allowed.append('corrective_complex')
        else:
            allowed = ['impulse_up', 'impulse_down', 'corrective_up',
                       'corrective_down', 'corrective_complex']
            forbidden = []

        return allowed, forbidden
