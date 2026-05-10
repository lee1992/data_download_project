from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# =============================================================================
# 全局配置 (所有可调参数集中在这里)
# =============================================================================
@dataclass
class SystemConfig:
    """
    ★★★ 所有可调参数在这里设置 ★★★

    调参指南:
    - zigzag_pct: 越大→识别越粗的摆动点, 小周期用小值(0.5-1%), 大周期用大值(3-5%)
    - dow_valid_swing_pct: 定义"有效回调"的最小幅度, 影响道氏趋势判定灵敏度
    - gann_snap_tolerance: 江恩吸附容差, 越大→越容易吸附, 一般1-3%
    - elliott_max_candidates: 波浪候选数量上限, 越大越慢但越全
    - mtf_resonance_threshold: 多周期共振阈值, 越高→信号越少但越可靠
    """

    # --- ZigZag参数 (按频率分别设置) ---
    zigzag_pct_by_freq: Dict[str, float] = field(default_factory=lambda: {
        '1min': 0.3,
        '5min': 0.5,
        '10min': 0.7,
        '15min': 1.0,
        '30min': 1.5,
        '60min': 2.0,
        '240min': 3.0,
        '1D': 3.0,
        '1W': 5.0,
        '1M': 8.0,
    })
    zigzag_default_pct: float = 2.0  # 未指定频率时的默认值

    # --- 道氏理论参数 ---
    dow_valid_swing_pct: float = 3.0  # 有效摆动的最小回调幅度(%)
    dow_trend_confirm_swings: int = 2  # 确认趋势所需的同向摆动点数
    dow_breakout_margin: float = 0.1  # 突破确认的余量(%), 防止假突破

    # --- 江恩理论参数 ---
    gann_price_ratios: List[float] = field(default_factory=lambda: [
        0.0, 0.125, 0.25, 1 / 3, 0.375, 0.5, 0.625, 2 / 3, 0.75, 0.875, 1.0,
        1.125, 1.25, 1.333, 1.375, 1.5, 1.618, 1.75, 2.0
    ])
    gann_time_ratios: List[float] = field(default_factory=lambda: [
        0.125, 0.25, 1 / 3, 0.375, 0.5, 0.618, 2 / 3, 0.75, 1.0,
        1.25, 1.333, 1.5, 1.618, 2.0, 2.618
    ])
    gann_snap_tolerance: float = 1.5  # 价格吸附容差(%)
    gann_time_snap_bars: int = 3  # 时间吸附容差(K线数)
    gann_resonance_weights: Dict[str, float] = field(default_factory=lambda: {
        'price_hit': 1.0,  # 价格落在江恩位上
        'time_hit': 1.0,  # 时间落在江恩窗口上
        'price_time_cross': 3.0  # 价格+时间同时命中(共振)
    })

    # --- 波浪理论参数 ---
    elliott_max_candidates: int = 10  # 最多保留的波浪候选结构数
    elliott_fib_ratios: List[float] = field(default_factory=lambda: [
        0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.0, 2.618
    ])
    elliott_w2_max_retrace: float = 0.99  # 第2浪最大回撤比例
    elliott_w4_w1_overlap: bool = False  # 是否允许4浪-1浪重叠(楔形时=True)

    # --- 多周期共振参数 ---
    # 定义周期层级 (从大到小)
    timeframe_hierarchy: List[str] = field(default_factory=lambda: [
        '1M', '1W', '1D', '240min', '60min', '30min', '15min', '10min', '5min', '1min'
    ])
    # 共振分析的周期对 (大周期, 小周期)
    resonance_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ('1W', '1D'),
        ('1D', '60min'),
        ('60min', '15min'),
        ('15min', '5min'),
    ])
    mtf_resonance_threshold: float = 0.6  # 共振评分阈值 (0-1)

    # --- 信号生成参数 ---
    signal_min_risk_reward: float = 2.0  # 最小盈亏比
    signal_stop_buffer_pct: float = 0.5  # 止损位额外缓冲(%)
