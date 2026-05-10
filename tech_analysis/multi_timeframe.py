import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from .config import SystemConfig
from .data_engine import DataEngine
from .dow_engine import DowEngine, DowConstraint, TrendDirection
from .gann_engine import GannEngine, GannGrid
from .elliott_engine import ElliottEngine, WaveCount

# =============================================================================
# Module 4: 多周期共振分析
# =============================================================================
@dataclass
class TimeframeAnalysis:
    """单一时间周期的分析结论"""
    freq: str
    dow_constraint: DowConstraint
    gann_grid: GannGrid
    wave_counts: List[WaveCount]
    primary_wave: Optional[WaveCount]  # 最优波浪计数
    current_position: str  # 当前处于哪个浪 (如 "推动3浪中" "调整B浪中")
    bias: str  # 偏向: 'bullish' / 'bearish' / 'neutral'
@dataclass
class ResonanceSignal:
    """多周期共振信号"""
    direction: str  # 'long' / 'short' / 'neutral'
    strength: float  # 共振强度 0-1
    large_tf: str  # 大周期
    small_tf: str  # 小周期
    large_bias: str  # 大周期偏向
    small_bias: str  # 小周期偏向
    large_position: str  # 大周期当前位置
    small_position: str  # 小周期当前位置
    entry_condition: str  # 入场条件描述
    invalidation: str  # 失效条件描述
    key_levels: Dict[str, float]  # 关键价格位 (止损/目标等)
class MultiTimeframeSync:
    """多周期共振分析器"""
    def __init__(self, config: SystemConfig):
        self.config = config
        self.data_engine = DataEngine(config)
        self.dow_engine = DowEngine(config)
        self.gann_engine = GannEngine(config)
        self.elliott_engine = ElliottEngine(config)
    def analyze_single_timeframe(self, df: pd.DataFrame,
                                 freq: str) -> TimeframeAnalysis:
        """对单一时间周期执行完整分析链"""
        # Step 1: ZigZag
        swings = self.data_engine.find_zigzag(df, freq=freq)
        if len(swings) < 4:
            return TimeframeAnalysis(
                freq=freq,
                dow_constraint=DowConstraint(
                    TrendDirection.NEUTRAL, TrendDirection.NEUTRAL,
                    [], [], 0, ['all'], [], 0.0
                ),
                gann_grid=GannGrid([], [], []),
                wave_counts=[],
                primary_wave=None,
                current_position="数据不足",
                bias='neutral'
            )
        current_price = df['close'].iloc[-1]
        # Step 2: 道氏约束
        dow = self.dow_engine.analyze(swings, current_price)
        # Step 3: 江恩网格
        gann = self.gann_engine.analyze(swings, df)
        # Step 4: 波浪枚举
        waves = self.elliott_engine.analyze(swings, dow, gann, current_price)
        # 确定主波浪和当前位置
        primary_wave = waves[0] if waves else None
        current_position = self._determine_position(primary_wave, current_price)
        bias = self._determine_bias(dow, primary_wave)
        return TimeframeAnalysis(
            freq=freq,
            dow_constraint=dow,
            gann_grid=gann,
            wave_counts=waves,
            primary_wave=primary_wave,
            current_position=current_position,
            bias=bias
        )
    def analyze_multi_timeframe(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        对多个时间周期执行分析并计算共振
        Parameters:
            data_dict: {freq: DataFrame} 各周期数据
        Returns:
            {
                'analyses': {freq: TimeframeAnalysis},
                'resonance_signals': [ResonanceSignal],
                'summary': str
            }
        """
        # 1. 逐周期分析
        analyses = {}
        for freq, df in data_dict.items():
            analyses[freq] = self.analyze_single_timeframe(df, freq)
        # 2. 计算共振信号
        signals = []
        for large_tf, small_tf in self.config.resonance_pairs:
            if large_tf in analyses and small_tf in analyses:
                signal = self._calc_resonance(
                    analyses[large_tf], analyses[small_tf],
                    large_tf, small_tf
                )
                if signal:
                    signals.append(signal)
        # 3. 生成综合摘要
        summary = self._generate_summary(analyses, signals)
        return {
            'analyses': analyses,
            'resonance_signals': signals,
            'summary': summary
        }
    def _determine_position(self, wave: Optional[WaveCount],
                            current_price: float) -> str:
        """判断当前价格处于波浪的哪个位置"""
        if wave is None:
            return "无法判定"
        # 找最后一个已完成的浪
        last_label = wave.labels[-1] if wave.labels else None
        if last_label is None:
            return "无法判定"
        wtype = wave.wave_type.value
        last_wave = last_label.wave_id
        if 'impulse' in wtype:
            if last_wave == '5':
                return f"推动{wtype.split('_')[1]}浪已完成, 等待调整"
            else:
                return f"推动浪第{last_wave}浪结束, 进入第{int(last_wave) + 1 if last_wave.isdigit() else '?'}浪"
        else:
            return f"调整浪{last_wave}浪结束"
    def _determine_bias(self, dow: DowConstraint,
                        wave: Optional[WaveCount]) -> str:
        """综合道氏和波浪确定偏向"""
        # 道氏偏向
        if dow.primary_trend == TrendDirection.UP:
            dow_bias = 'bullish'
        elif dow.primary_trend == TrendDirection.DOWN:
            dow_bias = 'bearish'
        else:
            dow_bias = 'neutral'
        # 波浪偏向
        if wave is None:
            return dow_bias
        wtype = wave.wave_type.value
        if 'impulse_up' in wtype or 'diagonal_up' in wtype:
            wave_bias = 'bullish'
        elif 'impulse_down' in wtype or 'diagonal_down' in wtype:
            wave_bias = 'bearish'
        else:
            wave_bias = 'neutral'
        # 一致 → 强信号, 不一致 → 中性
        if dow_bias == wave_bias:
            return dow_bias
        elif dow_bias == 'neutral':
            return wave_bias
        elif wave_bias == 'neutral':
            return dow_bias
        else:
            return 'neutral'  # 矛盾时保持中性
    def _calc_resonance(self, large: TimeframeAnalysis,
                        small: TimeframeAnalysis,
                        large_tf: str, small_tf: str) -> Optional[ResonanceSignal]:
        """计算两个周期之间的共振信号"""
        # 共振逻辑:
        # 大周期看涨 + 小周期看涨 → 做多信号
        # 大周期看跌 + 小周期看跌 → 做空信号
        # 不一致 → 无信号
        if large.bias == 'bullish' and small.bias == 'bullish':
            direction = 'long'
            strength = (large.dow_constraint.confidence +
                        (small.primary_wave.score / 100 if small.primary_wave else 0)) / 2
        elif large.bias == 'bearish' and small.bias == 'bearish':
            direction = 'short'
            strength = (large.dow_constraint.confidence +
                        (small.primary_wave.score / 100 if small.primary_wave else 0)) / 2
        else:
            direction = 'neutral'
            strength = 0.0
        if strength < self.config.mtf_resonance_threshold and direction != 'neutral':
            direction = 'neutral'  # 强度不够, 降级
        # 关键价格位
        key_levels = {}
        if large.dow_constraint.invalidation_price:
            key_levels['large_invalidation'] = large.dow_constraint.invalidation_price
        if small.dow_constraint.invalidation_price:
            key_levels['small_invalidation'] = small.dow_constraint.invalidation_price
        # 从江恩网格取最近的支撑/阻力
        if large.gann_grid.price_levels:
            sorted_levels = sorted(large.gann_grid.price_levels,
                                   key=lambda x: x.strength, reverse=True)
            if sorted_levels:
                key_levels['strong_gann_level'] = sorted_levels[0].value
        entry_condition = ""
        invalidation = ""
        if direction == 'long':
            entry_condition = f"小周期({small_tf})确认回调结束并突破前高时入场"
            invalidation = f"价格跌破{key_levels.get('small_invalidation', 'N/A')}时失效"
        elif direction == 'short':
            entry_condition = f"小周期({small_tf})确认反弹结束并跌破前低时入场"
            invalidation = f"价格突破{key_levels.get('small_invalidation', 'N/A')}时失效"
        return ResonanceSignal(
            direction=direction,
            strength=strength,
            large_tf=large_tf,
            small_tf=small_tf,
            large_bias=large.bias,
            small_bias=small.bias,
            large_position=large.current_position,
            small_position=small.current_position,
            entry_condition=entry_condition,
            invalidation=invalidation,
            key_levels=key_levels
        )
    def _generate_summary(self, analyses: dict, signals: list) -> str:
        """生成文字摘要"""
        lines = ["=" * 60, "多周期分析摘要", "=" * 60, ""]
        lines.append("【各周期结论】")
        for freq, analysis in sorted(analyses.items(),
                                     key=lambda x: self.config.timeframe_hierarchy.index(x[0])
                                     if x[0] in self.config.timeframe_hierarchy else 99):
            trend_str = analysis.dow_constraint.primary_trend.value
            wave_str = analysis.primary_wave.description if analysis.primary_wave else "无"
            score = f"{analysis.primary_wave.score:.1f}" if analysis.primary_wave else "N/A"
            lines.append(f"  {freq:>6s}: 道氏={trend_str} | 波浪={wave_str} | 评分={score} | 偏向={analysis.bias}")
        lines.append("")
        lines.append("【共振信号】")
        for sig in signals:
            emoji = "🟢" if sig.direction == 'long' else "🔴" if sig.direction == 'short' else "⚪"
            lines.append(f"  {emoji} {sig.large_tf}+{sig.small_tf}: {sig.direction} (强度={sig.strength:.2f})")
            if sig.entry_condition:
                lines.append(f"     入场: {sig.entry_condition}")
            if sig.invalidation:
                lines.append(f"     失效: {sig.invalidation}")
            if sig.key_levels:
                lines.append(f"     关键位: {sig.key_levels}")
        if not signals:
            lines.append("  无共振信号")
        return "\n".join(lines)
