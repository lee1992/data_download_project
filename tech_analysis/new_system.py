"""
=============================================================================
三理论串联约束交易系统 (Dow-Gann-Elliott Funnel Model)
=============================================================================
架构: 道氏(方向约束) → 江恩(坐标网格) → 波浪(结构枚举) → 多周期共振 → 信号
作者: Assistant
说明: 回测模块预留接口,核心分析链完整可运行
=============================================================================
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# 导入拆分后的子模块
from tech_analysis.config import SystemConfig
from tech_analysis.data_engine import DataEngine
from tech_analysis.dow_engine import TrendDirection, DowConstraint, DowEngine
from tech_analysis.gann_engine import GannLevel, GannGrid, GannEngine
from tech_analysis.elliott_engine import WaveType, WaveLabel, WaveCount, ElliottEngine
from tech_analysis.multi_timeframe import TimeframeAnalysis, ResonanceSignal, MultiTimeframeSync
from tech_analysis.signal_generator import TradeSignal, SignalGenerator, BacktestEngine

# =============================================================================
# 主流程: 一键运行
# =============================================================================
class TradingSystem:
    """三理论串联约束交易系统 - 主入口"""
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.data_engine = DataEngine(self.config)
        self.mtf_sync = MultiTimeframeSync(self.config)
        self.signal_gen = SignalGenerator(self.config)
    def run(self, raw_df: pd.DataFrame,
            source_freq: str,
            target_freqs: List[str] = None,
            resonance_pairs: List[Tuple[str, str]] = None) -> Dict:
        """
        完整运行流程
        Parameters:
            raw_df: 原始OHLCV数据 (DatetimeIndex, columns: open,high,low,close,volume)
            source_freq: 原始数据频率 (如 '1min', '5min', '1D')
            target_freqs: 需要分析的目标频率列表 (None=使用配置默认值)
            resonance_pairs: 自定义共振对 (None=使用配置默认值)
        Returns:
            {
                'data': {freq: DataFrame},
                'analyses': {freq: TimeframeAnalysis},
                'resonance_signals': [ResonanceSignal],
                'trade_signals': [TradeSignal],
                'summary': str
            }
        """
        if resonance_pairs:
            self.config.resonance_pairs = resonance_pairs
        # 1. 确定目标频率
        if target_freqs is None:
            # 只生成比source_freq更低频的
            hierarchy = self.config.timeframe_hierarchy
            if source_freq in hierarchy:
                src_idx = hierarchy.index(source_freq)
                target_freqs = hierarchy[:src_idx + 1]  # 包含自身和更高周期
            else:
                target_freqs = [source_freq]
        # 2. 重采样
        data_dict = self.data_engine.load_and_resample(raw_df, source_freq, target_freqs)
        # 3. 多周期分析
        mtf_result = self.mtf_sync.analyze_multi_timeframe(data_dict)
        # 4. 生成信号
        trade_signals = self.signal_gen.generate(mtf_result)
        # 5. 输出
        result = {
            'data': data_dict,
            'analyses': mtf_result['analyses'],
            'resonance_signals': mtf_result['resonance_signals'],
            'trade_signals': trade_signals,
            'summary': mtf_result['summary']
        }
        # 打印摘要
        print(result['summary'])
        if trade_signals:
            print("\n" + "=" * 60)
            print("交易信号")
            print("=" * 60)
            for ts in trade_signals:
                print(f"  {'🟢做多' if ts.direction == 'long' else '🔴做空'} "
                      f"入场={ts.entry_price} 止损={ts.stop_loss} "
                      f"止盈={ts.take_profit} 盈亏比={ts.risk_reward} "
                      f"来源={ts.source}")
        return result

# =============================================================================
# 使用示例
# =============================================================================
if __name__ == "__main__":
    # ---- 生成模拟数据用于测试 ----
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=5000, freq='5min')
    # 模拟一个有趋势的价格序列
    returns = np.random.randn(5000) * 0.002 + 0.0001  # 微弱上涨趋势
    price = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        'open': price * (1 + np.random.randn(5000) * 0.001),
        'high': price * (1 + abs(np.random.randn(5000)) * 0.003),
        'low': price * (1 - abs(np.random.randn(5000)) * 0.003),
        'close': price,
        'volume': np.random.randint(1000, 10000, 5000)
    }, index=dates)
    filename="SH_000001_K_5M_2017-04-22_2026-04-28.csv"
    df2 = pd.read_csv(f'.\\futu_data\\K_5M\\{filename}')
    #df2 =  df2.tail(1000).copy()
    df2.set_index('time_key',inplace=True)
    df = df2[['open', 'high', 'low', 'close', 'volume']].copy()
    # ---- 配置 (可调参数) ----
    config = SystemConfig()
    # ★ 自定义参数示例:
    # config.zigzag_pct_by_freq['5min'] = 0.8     # 调整5分钟ZigZag灵敏度
    # config.dow_valid_swing_pct = 2.0              # 调整道氏有效摆动阈值
    # config.gann_snap_tolerance = 2.0              # 调整江恩吸附容差
    # config.elliott_max_candidates = 5             # 减少波浪候选数
    # config.mtf_resonance_threshold = 0.5          # 降低共振阈值(更多信号)
    # ★ 自定义共振对:
    # config.resonance_pairs = [('1D', '60min'), ('60min', '5min')]
    # ---- 运行 ----
    system = TradingSystem(config)
    result = system.run(
        raw_df=df,
        source_freq='5min',
        target_freqs=['5min', '15min', '60min', '1D'],
        resonance_pairs=[('1D', '60min'), ('60min', '5min')]
    )
    import json
    import pandas as pd
    import numpy as np
    from enum import Enum


    # 扩展的JSON编码器，自动处理你系统里的所有特殊类型
    class TechAnalysisJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            # 1. 处理枚举类型 (比如你的WaveType)
            if isinstance(obj, Enum):
                return obj.value
            # 2. 处理pandas的时间戳
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            # 3. 处理DataFrame
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            # 4. 处理numpy的整数
            if isinstance(obj, np.integer):
                return int(obj)
            # 5. 处理numpy的浮点数
            if isinstance(obj, np.floating):
                return float(obj)
            # 6. 处理numpy数组
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # 7. 处理你所有的自定义dataclass (WaveCount、WaveLabel、GannGrid、GannLevel等)
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            # 其他类型交给默认处理
            return super().default(obj)


    # ========== 你的保存代码 ==========
    # 运行完你的分析得到result之后
    output_path = f'.\\tech_analysis\\result\\multi_timeframe_result.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result['analyses'], f, ensure_ascii=False, indent=4, cls=TechAnalysisJSONEncoder)

    # ---- 访问详细结果 ----
    #print(    result['analyses']['1D'].dow_constraint,result['analyses']['1D'].gann_grid ,    result['analyses']['5min'].wave_counts,result['resonance_signals'],result['trade_signals']    )
    # result['analyses']['1D'].dow_constraint        # 日线道氏约束
    # result['analyses']['60min'].gann_grid           # 60分钟江恩网格
    # result['analyses']['5min'].wave_counts          # 5分钟波浪候选
    # result['resonance_signals']                     # 共振信号
    # result['trade_signals']                         # 交易信号
