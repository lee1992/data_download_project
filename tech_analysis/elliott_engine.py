import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tech_analysis.config import SystemConfig
from tech_analysis.dow_engine import DowConstraint
from tech_analysis.gann_engine import GannGrid,GannLevel

class WaveType(Enum):
    IMPULSE_UP = "impulse_up"
    IMPULSE_DOWN = "impulse_down"
    CORRECTIVE_ZIGZAG = "corrective_zigzag"
    CORRECTIVE_FLAT = "corrective_flat"
    CORRECTIVE_TRIANGLE = "corrective_triangle"
    CORRECTIVE_COMPLEX = "corrective_complex"
    DIAGONAL_UP = "diagonal_up"
    DIAGONAL_DOWN = "diagonal_down"


@dataclass
class WaveLabel:
    """一个波浪标注"""
    wave_id: str  # 如 "1", "2", "3", "4", "5", "A", "B", "C"
    start_idx: int  # 起点的摆动点索引
    end_idx: int  # 终点的摆动点索引
    start_price: float
    end_price: float
    start_time: str
    end_time: str
    degree: int = 0  # 层级 (0=当前, 1=上一级, -1=下一级)


@dataclass
class WaveCount:
    """一套完整的波浪计数方案"""
    wave_type: WaveType  # 整体结构类型
    labels: List[WaveLabel]  # 各浪标注
    score: float = 0.0  # 综合评分 (0-100)
    hard_rule_pass: bool = True  # 是否通过硬约束
    dow_consistent: bool = True  # 是否与道氏约束一致
    gann_snap_score: float = 0.0  # 江恩吸附度评分
    guideline_score: float = 0.0  # 指南符合度评分
    description: str = ""  # 文字描述


class ElliottEngine:
    """波浪理论引擎: 在约束空间内枚举结构"""

    def __init__(self, config: SystemConfig):
        self.config = config

    def analyze(self, swings: pd.DataFrame,
                dow_constraint: DowConstraint,
                gann_grid: GannGrid,
                current_price: float) -> List[WaveCount]:
        """
        在道氏约束+江恩网格约束下, 枚举波浪计数候选

        Returns:
            List[WaveCount] 按评分排序的候选结构
        """
        candidates = []

        if len(swings) < 5:
            return candidates

        prices = swings['price'].values
        types = swings['type'].values
        dts = [str(d) for d in swings['datetime'].values]
        bar_indices = swings['bar_index'].values

        # 1. 尝试5浪推动结构
        impulse_candidates = self._try_impulse(
            prices, types, dts, bar_indices, dow_constraint, gann_grid
        )
        candidates.extend(impulse_candidates)

        # 2. 尝试3浪调整结构 (A-B-C)
        corrective_candidates = self._try_corrective(
            prices, types, dts, bar_indices, dow_constraint, gann_grid
        )
        candidates.extend(corrective_candidates)

        # 3. 过滤: 移除被道氏约束禁止的结构
        candidates = [c for c in candidates
                      if c.wave_type.value not in dow_constraint.forbidden_wave_types]

        # 4. 评分排序
        for c in candidates:
            c.score = self._calc_total_score(c)

        candidates.sort(key=lambda x: x.score, reverse=True)

        return candidates[:self.config.elliott_max_candidates]

    def _try_impulse(self, prices, types, dts, bar_indices,
                     dow: DowConstraint, gann: GannGrid) -> List[WaveCount]:
        """尝试标注5浪推动结构"""
        candidates = []
        n = len(prices)

        if n < 6:  # 至少需要6个摆动点 (起点+5个浪终点)
            return candidates

        # 枚举可能的5浪起点
        # 策略: 从最近的摆动点往回找, 限制搜索深度
        max_lookback = min(n, 20)
        start_range = range(max(0, n - max_lookback), n - 5)

        for start in start_range:
            # 5浪推动需要: 起点→1终→2终→3终→4终→5终
            # 在start之后的摆动点中选5个作为各浪终点
            remaining = list(range(start + 1, n))

            if len(remaining) < 5:
                continue

            # 简化: 连续取5个点 (也可以用组合枚举, 但太慢)
            for offset in range(len(remaining) - 4):
                indices = [start] + remaining[offset:offset + 5]

                if len(indices) != 6:
                    continue

                p = [prices[i] for i in indices]  # [起点, 1终, 2终, 3终, 4终, 5终]

                # 判断方向
                is_up = p[5] > p[0]

                if is_up:
                    wtype = WaveType.IMPULSE_UP
                    # 硬约束检查
                    if not self._check_impulse_rules_up(p):
                        continue
                else:
                    wtype = WaveType.IMPULSE_DOWN
                    if not self._check_impulse_rules_down(p):
                        continue

                # 构建WaveCount
                labels = []
                wave_names = ['1', '2', '3', '4', '5']
                for w in range(5):
                    labels.append(WaveLabel(
                        wave_id=wave_names[w],
                        start_idx=indices[w],
                        end_idx=indices[w + 1],
                        start_price=prices[indices[w]],
                        end_price=prices[indices[w + 1]],
                        start_time=dts[indices[w]],
                        end_time=dts[indices[w + 1]]
                    ))

                # 道氏一致性
                dow_ok = wtype.value in dow.allowed_wave_types

                # 江恩吸附度
                gann_score = self._calc_gann_snap(
                    [prices[i] for i in indices], gann.price_levels
                )

                # 指南评分
                guide_score = self._calc_guideline_score_impulse(p, is_up)

                wc = WaveCount(
                    wave_type=wtype,
                    labels=labels,
                    hard_rule_pass=True,
                    dow_consistent=dow_ok,
                    gann_snap_score=gann_score,
                    guideline_score=guide_score,
                    description=f"{'上涨' if is_up else '下跌'}推动浪 起点bar{indices[0]}"
                )
                candidates.append(wc)

        return candidates

    def _check_impulse_rules_up(self, p: list) -> bool:
        """检查上涨推动浪的3条铁律"""
        # p = [起点, 1终, 2终, 3终, 4终, 5终]
        w0, w1, w2, w3, w4, w5 = p

        # 基本方向: 1浪上, 2浪下, 3浪上, 4浪下, 5浪上
        if not (w1 > w0 and w2 < w1 and w3 > w2 and w4 < w3 and w5 > w4):
            # 允许5浪失败 (truncation): w5 可以 < w3 但必须 > w4
            if not (w1 > w0 and w2 < w1 and w3 > w2 and w4 < w3 and w5 > w4):
                return False

        # 规则1: 2浪不能回撤超过1浪的100%
        if w2 <= w0:
            return False

        # 规则2: 3浪不能是最短的推动浪
        wave1_len = w1 - w0
        wave3_len = w3 - w2
        wave5_len = w5 - w4
        if wave3_len < wave1_len and wave3_len < wave5_len:
            return False

        # 规则3: 4浪不能进入1浪价格区间
        if not self.config.elliott_w4_w1_overlap:
            if w4 < w1:
                return False

        return True

    def _check_impulse_rules_down(self, p: list) -> bool:
        """检查下跌推动浪的3条铁律"""
        w0, w1, w2, w3, w4, w5 = p

        if not (w1 < w0 and w2 > w1 and w3 < w2 and w4 > w3 and w5 < w4):
            return False

        if w2 >= w0:
            return False

        wave1_len = abs(w1 - w0)
        wave3_len = abs(w3 - w2)
        wave5_len = abs(w5 - w4)
        if wave3_len < wave1_len and wave3_len < wave5_len:
            return False

        if not self.config.elliott_w4_w1_overlap:
            if w4 > w1:
                return False

        return True

    def _try_corrective(self, prices, types, dts, bar_indices,
                        dow: DowConstraint, gann: GannGrid) -> List[WaveCount]:
        """尝试标注A-B-C调整结构"""
        candidates = []
        n = len(prices)

        if n < 4:  # 至少4个点 (起点+A+B+C)
            return candidates

        max_lookback = min(n, 15)
        start_range = range(max(0, n - max_lookback), n - 3)

        for start in start_range:
            remaining = list(range(start + 1, n))

            for offset in range(len(remaining) - 2):
                indices = [start] + remaining[offset:offset + 3]

                if len(indices) != 4:
                    continue

                p = [prices[i] for i in indices]
                w0, wA, wB, wC = p

                # Zigzag下跌: A下 B上 C下
                if wA < w0 and wB > wA and wC < wB:
                    # B浪不能超过起点
                    if wB > w0:
                        wtype = WaveType.CORRECTIVE_FLAT  # 扩大平坦型
                    else:
                        wtype = WaveType.CORRECTIVE_ZIGZAG
                    is_down_correction = True

                # Zigzag上涨: A上 B下 C上
                elif wA > w0 and wB < wA and wC > wB:
                    wtype = WaveType.CORRECTIVE_ZIGZAG
                    is_down_correction = False
                else:
                    continue

                labels = []
                for w, name in enumerate(['A', 'B', 'C']):
                    labels.append(WaveLabel(
                        wave_id=name,
                        start_idx=indices[w],
                        end_idx=indices[w + 1],
                        start_price=prices[indices[w]],
                        end_price=prices[indices[w + 1]],
                        start_time=dts[indices[w]],
                        end_time=dts[indices[w + 1]]
                    ))

                dow_ok = True  # 调整浪通常与两个方向都兼容
                gann_score = self._calc_gann_snap(
                    [prices[i] for i in indices], gann.price_levels
                )

                wc = WaveCount(
                    wave_type=wtype,
                    labels=labels,
                    hard_rule_pass=True,
                    dow_consistent=dow_ok,
                    gann_snap_score=gann_score,
                    guideline_score=self._calc_guideline_score_corrective(p),
                    description=f"{'下跌' if is_down_correction else '上涨'}调整浪({wtype.value}) 起点bar{indices[0]}"
                )
                candidates.append(wc)

        return candidates

    def _calc_gann_snap(self, node_prices: list,
                        gann_levels: List[GannLevel]) -> float:
        """计算波浪节点与江恩位的吸附度评分"""
        if not gann_levels:
            return 0.0

        tol = self.config.gann_snap_tolerance / 100.0
        total_snap = 0

        for price in node_prices:
            best_snap = 0
            for lvl in gann_levels:
                dist = abs(price - lvl.value) / max(price, 1e-10)
                if dist < tol:
                    snap = (1 - dist / tol) * lvl.strength
                    best_snap = max(best_snap, snap)
            total_snap += best_snap

        return total_snap / len(node_prices) * 100  # 归一化到0-100

    def _calc_guideline_score_impulse(self, p: list, is_up: bool) -> float:
        """推动浪的指南评分"""
        score = 0
        w0, w1, w2, w3, w4, w5 = p

        if is_up:
            wave1 = w1 - w0
            wave2_retrace = (w1 - w2) / wave1 if wave1 > 0 else 0
            wave3 = w3 - w2
            wave4_retrace = (w3 - w4) / wave3 if wave3 > 0 else 0
            wave5 = w5 - w4

            # 指南: 2浪回撤通常在38.2%-61.8%
            if 0.382 <= wave2_retrace <= 0.618:
                score += 20
            elif 0.236 <= wave2_retrace <= 0.786:
                score += 10

            # 指南: 3浪通常是1浪的1.618倍或更长
            if wave1 > 0:
                w3_ratio = wave3 / wave1
                if 1.5 <= w3_ratio <= 1.75:
                    score += 25
                elif 1.0 <= w3_ratio <= 2.618:
                    score += 15

            # 指南: 4浪回撤通常在23.6%-38.2%
            if 0.236 <= wave4_retrace <= 0.382:
                score += 20
            elif 0.1 <= wave4_retrace <= 0.5:
                score += 10

            # 指南: 交替原则 (2浪和4浪形态应不同, 简化为回撤比例差异)
            if abs(wave2_retrace - wave4_retrace) > 0.15:
                score += 15

            # 指南: 3浪最长加分
            if wave3 > wave1 and wave3 > wave5:
                score += 20
        else:
            # 下跌推动浪, 镜像处理
            score = self._calc_guideline_score_impulse(
                [-x for x in p], True
            )

        return min(score, 100)

    def _calc_guideline_score_corrective(self, p: list) -> float:
        """调整浪的指南评分"""
        score = 50  # 基础分

        w0, wA, wB, wC = p
        a_len = abs(wA - w0)

        if a_len > 0:
            # C浪通常等于A浪或是A浪的1.618倍
            c_len = abs(wC - wB)
            ratio = c_len / a_len

            if 0.9 <= ratio <= 1.1:
                score += 30  # C = A
            elif 1.5 <= ratio <= 1.75:
                score += 25  # C = 1.618 × A
            elif 0.5 <= ratio <= 0.7:
                score += 15  # C = 0.618 × A

            # B浪回撤
            b_retrace = abs(wB - wA) / a_len
            if 0.382 <= b_retrace <= 0.618:
                score += 20

        return min(score, 100)

    def _calc_total_score(self, wc: WaveCount) -> float:
        """计算综合评分"""
        score = 0

        # 硬约束通过: 必须条件
        if not wc.hard_rule_pass:
            return 0

        # 道氏一致性: 权重30%
        score += 30 if wc.dow_consistent else 0

        # 江恩吸附度: 权重30%
        score += wc.gann_snap_score * 0.3

        # 指南评分: 权重40%
        score += wc.guideline_score * 0.4

        return score