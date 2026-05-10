import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from .config import SystemConfig
from .multi_timeframe import ResonanceSignal

# =============================================================================
# Module 5: 信号生成器 (含回测接口预留)
# =============================================================================
@dataclass
class TradeSignal:
    """交易信号"""
    timestamp: str
    direction: str  # 'long' / 'short'
    entry_price: float  # 建议入场价
    stop_loss: float  # 止损价
    take_profit: float  # 止盈价
    risk_reward: float  # 盈亏比
    strength: float  # 信号强度
    source: str  # 信号来源描述
    conditions: Dict  # 触发/失效条件
class SignalGenerator:
    """信号生成器"""
    def __init__(self, config: SystemConfig):
        self.config = config
    def generate(self, mtf_result: Dict) -> List[TradeSignal]:
        """从多周期分析结果生成交易信号"""
        signals = []
        for res_signal in mtf_result.get('resonance_signals', []):
            if res_signal.direction == 'neutral':
                continue
            trade_signal = self._convert_to_trade_signal(res_signal, mtf_result)
            if trade_signal and trade_signal.risk_reward >= self.config.signal_min_risk_reward:
                signals.append(trade_signal)
        return signals
    def _convert_to_trade_signal(self, res: ResonanceSignal,
                                 mtf_result: Dict) -> Optional[TradeSignal]:
        """将共振信号转换为交易信号"""
        # 从小周期数据取当前价格
        small_analysis = mtf_result['analyses'].get(res.small_tf)
        if small_analysis is None:
            return None
        # 简化: 用道氏失效价作为止损
        invalidation = res.key_levels.get('small_invalidation', 0)
        gann_level = res.key_levels.get('strong_gann_level', 0)
        if invalidation == 0:
            return None
        # 估算入场价 (用最新价的近似)
        if small_analysis.primary_wave and small_analysis.primary_wave.labels:
            entry_price = small_analysis.primary_wave.labels[-1].end_price
        else:
            return None
        buffer = self.config.signal_stop_buffer_pct / 100.0
        if res.direction == 'long':
            stop_loss = invalidation * (1 - buffer)
            risk = entry_price - stop_loss
            if risk <= 0:
                return None
            take_profit = entry_price + risk * self.config.signal_min_risk_reward
            if gann_level > entry_price:
                take_profit = max(take_profit, gann_level)
        else:
            stop_loss = invalidation * (1 + buffer)
            risk = stop_loss - entry_price
            if risk <= 0:
                return None
            take_profit = entry_price - risk * self.config.signal_min_risk_reward
            if gann_level < entry_price and gann_level > 0:
                take_profit = min(take_profit, gann_level)
        rr = abs(entry_price - take_profit) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
        return TradeSignal(
            timestamp=str(pd.Timestamp.now()),
            direction=res.direction,
            entry_price=round(entry_price, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            risk_reward=round(rr, 2),
            strength=res.strength,
            source=f"{res.large_tf}+{res.small_tf}共振",
            conditions={
                'entry': res.entry_condition,
                'invalidation': res.invalidation,
                'large_trend': res.large_bias,
                'small_trend': res.small_bias,
            }
        )
# =============================================================================
# 回测接口预留
# =============================================================================
class BacktestEngine:
    """回测引擎 (接口预留, 后续实现)"""
    def __init__(self, config: SystemConfig):
        self.config = config
    def run(self, df: pd.DataFrame, signals: List[TradeSignal]) -> Dict:
        """
        TODO: 实现回测逻辑
        Parameters:
            df: 用于回测的OHLCV数据
            signals: 交易信号列表
        Returns:
            回测绩效报告
        """
        raise NotImplementedError("回测引擎待实现")
    def report(self, results: Dict) -> str:
        """TODO: 生成绩效报告"""
        raise NotImplementedError
