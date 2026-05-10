import pandas as pd
from typing import List, Dict
from tech_analysis.config import SystemConfig


# =============================================================================
# Module 0: 数据引擎 (重采样 + 跳空处理 + ZigZag)
# =============================================================================

class DataEngine:
    """数据预处理: 重采样、跳空标记、ZigZag摆动点识别"""

    # 重采样映射 (pandas offset aliases)
    RESAMPLE_MAP = {
        '1min': '1min', '5min': '5min', '10min': '10min',
        '15min': '15min', '30min': '30min', '60min': '60min',
        '240min': '4h',
        '1D': '1D', '1W': '1W', '1M': '1ME',
    }

    def __init__(self, config: SystemConfig):
        self.config = config

    def load_and_resample(self, df: pd.DataFrame,
                          source_freq: str,
                          target_freqs: List[str]) -> Dict[str, pd.DataFrame]:
        """
        从原始数据重采样到多个目标频率

        Parameters:
            df: OHLCV DataFrame, index为DatetimeIndex, 列: open,high,low,close,volume
            source_freq: 原始数据频率 (如 '1min')
            target_freqs: 需要生成的目标频率列表

        Returns:
            {freq_str: DataFrame} 字典
        """
        results = {}

        # 确保index是datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 标准化列名
        df.columns = [c.lower().strip() for c in df.columns]

        for freq in target_freqs:
            if freq == source_freq:
                resampled = df.copy()
            else:
                offset = self.RESAMPLE_MAP.get(freq)
                if offset is None:
                    raise ValueError(f"不支持的频率: {freq}")

                resampled = df.resample(offset).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()

            # 标记跳空
            resampled = self._mark_gaps(resampled)
            results[freq] = resampled

        return results

    def _mark_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """标记跳空缺口"""
        df = df.copy()
        prev_close = df['close'].shift(1)
        df['gap_up'] = df['open'] > prev_close * 1.001  # 向上跳空 > 0.1%
        df['gap_down'] = df['open'] < prev_close * 0.999  # 向下跳空
        df['gap_size'] = (df['open'] - prev_close) / prev_close * 100  # 跳空幅度%
        return df

    def find_zigzag(self, df: pd.DataFrame, freq: str = None,
                    pct: float = None) -> pd.DataFrame:
        """
        ZigZag摆动点识别

        Parameters:
            df: OHLCV DataFrame
            freq: 频率标识, 用于查找对应的zigzag参数
            pct: 直接指定百分比阈值 (优先于freq查找)

        Returns:
            DataFrame with columns: [datetime, price, type('H'/'L'), bar_index]
        """
        if pct is None:
            pct = self.config.zigzag_pct_by_freq.get(
                freq, self.config.zigzag_default_pct
            )

        threshold = pct / 100.0
        highs = df['high'].values
        lows = df['low'].values
        dates = df.index

        swings = []  # (bar_index, price, type)

        if len(highs) < 2:
            return pd.DataFrame(columns=['datetime', 'price', 'type', 'bar_index'])

        # 初始化: 找第一个方向
        last_high_idx, last_high = 0, highs[0]
        last_low_idx, last_low = 0, lows[0]
        direction = 0  # 0=未定, 1=找高点, -1=找低点

        for i in range(1, len(highs)):
            if direction == 0:
                # 确定初始方向
                if highs[i] >= last_high * (1 + threshold):
                    # 上涨启动
                    swings.append((last_low_idx, last_low, 'L'))
                    last_high_idx, last_high = i, highs[i]
                    direction = 1
                elif lows[i] <= last_low * (1 - threshold):
                    # 下跌启动
                    swings.append((last_high_idx, last_high, 'H'))
                    last_low_idx, last_low = i, lows[i]
                    direction = -1
                else:
                    if highs[i] > last_high:
                        last_high_idx, last_high = i, highs[i]
                    if lows[i] < last_low:
                        last_low_idx, last_low = i, lows[i]

            elif direction == 1:
                # 当前在找高点(上涨中)
                if highs[i] > last_high:
                    last_high_idx, last_high = i, highs[i]
                elif lows[i] <= last_high * (1 - threshold):
                    # 反转向下, 确认高点
                    swings.append((last_high_idx, last_high, 'H'))
                    last_low_idx, last_low = i, lows[i]
                    direction = -1

            elif direction == -1:
                # 当前在找低点(下跌中)
                if lows[i] < last_low:
                    last_low_idx, last_low = i, lows[i]
                elif highs[i] >= last_low * (1 + threshold):
                    # 反转向上, 确认低点
                    swings.append((last_low_idx, last_low, 'L'))
                    last_high_idx, last_high = i, highs[i]
                    direction = 1

        # 添加最后一个点
        if direction == 1:
            swings.append((last_high_idx, last_high, 'H'))
        elif direction == -1:
            swings.append((last_low_idx, last_low, 'L'))

        # 构建DataFrame
        swing_df = pd.DataFrame(swings, columns=['bar_index', 'price', 'type'])
        swing_df['datetime'] = [dates[i] for i in swing_df['bar_index']]

        return swing_df[['datetime', 'price', 'type', 'bar_index']]