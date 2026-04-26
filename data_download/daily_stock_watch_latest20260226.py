"""
富途OpenAPI 股票筛选工具
功能：
1. 读取自选股列表 (list_part1)
2. 合并用户自定义列表 (list_part2)
3. 获取近2个月日线数据 (开盘价、收盘价、最高价、最低价)
4. 根据自定义规则筛选股票（KDJ/SlowKD 4个条件 + RSI 4个条件）

使用前提：
1. 安装 moomoo-api: pip install moomoo-api
2. 启动 OpenD 网关程序
"""

from futu import *
import pandas as pd
from datetime import datetime, timedelta
import time

# ==================== 配置参数 ====================
CONFIG = {
    'host': '127.0.0.1',  # OpenD 地址
    'port': 11111,  # OpenD 端口
    'watchlist_group': '港股',  # 自选股分组名称，可修改为你的分组名
    'batch_size': 60,  # 每批请求的股票数量（避免频率限制）
    'batch_interval': 61.0,  # 每批之间的间隔秒数
    'kline_days': 90,  # 获取近N天的K线数据（需要足够数据计算KDJ）
}

# ==================== 可调参数：自定义股票列表 ====================
list_part2 = [
    'SH.510050',  # 腾讯控股
    'SH.510300',  # 阿里巴巴
    'SH.510500',
    "SZ.159915",
    "SH.588000"
    # 添加更多股票代码...
]


# ==================== KDJ/SlowKD 计算辅助函数 ====================
def calculate_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """
    计算KDJ指标（参数9,3,3）
    标准KDJ公式：
    - RSV = (Close - Low_n) / (High_n - Low_n) * 100
    - K = SMA(RSV, m1)  即 K = K(-1) * (m1-1)/m1 + RSV * 1/m1
    - D = SMA(K, m2)    即 D = D(-1) * (m2-1)/m2 + K * 1/m2
    - J = 3*K - 2*D

    :param df: 包含high/low/close/time_key的K线DataFrame
    :param n: RSV周期（默认9）
    :param m1: K值平滑周期（默认3）
    :param m2: D值平滑周期（默认3）
    :return: 新增kdj_k/kdj_d/kdj_j列的DataFrame
    """
    df = df.copy()
    df = df.sort_values('time_key').reset_index(drop=True)

    # 1. 计算N日内最低价和最高价
    df['low_n'] = df['low'].rolling(window=n, min_periods=1).min()
    df['high_n'] = df['high'].rolling(window=n, min_periods=1).max()

    # 2. 计算RSV
    df['rsv'] = 0.0
    mask = (df['high_n'] - df['low_n']) != 0
    df.loc[mask, 'rsv'] = (df.loc[mask, 'close'] - df.loc[mask, 'low_n']) / \
                          (df.loc[mask, 'high_n'] - df.loc[mask, 'low_n']) * 100

    # 3. 计算K值和D值（使用SMA平滑，初始值50）
    k_values = [50.0]  # K初始值
    d_values = [50.0]  # D初始值

    for i in range(1, len(df)):
        rsv = df.loc[df.index[i], 'rsv']
        # K = K(-1) * (m1-1)/m1 + RSV * 1/m1
        new_k = k_values[-1] * (m1 - 1) / m1 + rsv / m1
        k_values.append(new_k)
        # D = D(-1) * (m2-1)/m2 + K * 1/m2
        new_d = d_values[-1] * (m2 - 1) / m2 + new_k / m2
        d_values.append(new_d)

    df['kdj_k'] = k_values
    df['kdj_d'] = d_values
    # J = 3K - 2D
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    # 清理临时列
    df.drop(columns=['low_n', 'high_n', 'rsv'], inplace=True, errors='ignore')

    return df


def calculate_slowkd(df: pd.DataFrame, n: int = 9, m: int = 3) -> pd.DataFrame:
    """
    计算SlowKD指标（参数9,3）
    SlowKD公式：
    - FastK = 标准KDJ的K值
    - SlowK = SMA(FastK, m)
    - SlowD = SMA(SlowK, m)

    :param df: 包含high/low/close/time_key的K线DataFrame
    :param n: 快速K周期（默认9）
    :param m: 慢速平滑周期（默认3）
    :return: 新增slowkd_k/slowkd_d列的DataFrame
    """
    df = df.copy()
    df = df.sort_values('time_key').reset_index(drop=True)

    # 1. 计算N日内最低价和最高价
    df['low_n'] = df['low'].rolling(window=n, min_periods=1).min()
    df['high_n'] = df['high'].rolling(window=n, min_periods=1).max()

    # 2. 计算RSV
    df['rsv'] = 0.0
    mask = (df['high_n'] - df['low_n']) != 0
    df.loc[mask, 'rsv'] = (df.loc[mask, 'close'] - df.loc[mask, 'low_n']) / \
                          (df.loc[mask, 'high_n'] - df.loc[mask, 'low_n']) * 100

    # 3. 计算FastK（即标准KDJ的K值）
    fastk_values = [50.0]
    for i in range(1, len(df)):
        rsv = df.loc[df.index[i], 'rsv']
        new_fastk = fastk_values[-1] * 2 / 3 + rsv / 3
        fastk_values.append(new_fastk)
    df['fastk'] = fastk_values

    # 4. 计算SlowK = SMA(FastK, m)
    slowk_values = [50.0]
    for i in range(1, len(df)):
        fastk = df.loc[df.index[i], 'fastk']
        new_slowk = slowk_values[-1] * (m - 1) / m + fastk / m
        slowk_values.append(new_slowk)
    df['slowkd_k'] = slowk_values

    # 5. 计算SlowD = SMA(SlowK, m)
    slowd_values = [50.0]
    for i in range(1, len(df)):
        slowk = df.loc[df.index[i], 'slowkd_k']
        new_slowd = slowd_values[-1] * (m - 1) / m + slowk / m
        slowd_values.append(new_slowd)
    df['slowkd_d'] = slowd_values

    # 清理临时列
    df.drop(columns=['low_n', 'high_n', 'rsv', 'fastk'], inplace=True, errors='ignore')

    return df


# ==================== RSI 计算辅助函数（新增） ====================
def calculate_rsi(df: pd.DataFrame, period: int = 6) -> pd.DataFrame:
    """
    计算RSI指标
    RSI公式：
    - RSI = 100 - 100 / (1 + RS)
    - RS = 平均上涨幅度 / 平均下跌幅度
    使用EMA（指数移动平均）计算平均涨跌幅

    :param df: 包含close/time_key的K线DataFrame
    :param period: RSI周期（默认6）
    :return: 新增rsi_{period}列的DataFrame
    """
    df = df.copy()
    df = df.sort_values('time_key').reset_index(drop=True)

    # 计算价格变化
    delta = df['close'].diff()

    # 分离上涨和下跌
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # 使用EMA计算平均涨跌幅
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # 计算RS和RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # 处理除零情况
    rsi = rsi.fillna(50)  # 初始值设为50
    rsi = rsi.replace([float('inf'), float('-inf')], [100, 0])

    df[f'rsi_{period}'] = rsi

    return df


def get_last_month_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    从K线数据中筛选出「前1个月（30天）」的数据
    :param df: 原始K线DataFrame
    :return: 近30天的K线数据
    """
    df = df.copy()
    df['time_key'] = pd.to_datetime(df['time_key'])
    last_month_date = datetime.now() - timedelta(days=30)
    last_month_df = df[df['time_key'] >= last_month_date].copy()
    return last_month_df


# ==================== 4个独立筛选规则 ====================
def custom_filter_rule_kdj1(code: str, name: str, kline_df: pd.DataFrame) -> bool:
    """
    规则1：前1个月内出现过KDJ K值低于20
    KDJ参数：9,3,3
    """
    if kline_df.empty or len(kline_df) < 20:
        return False

    df = calculate_kdj(kline_df, n=9, m1=3, m2=3)
    last_month_df = get_last_month_data(df)

    if last_month_df.empty:
        return False

    return (last_month_df['kdj_k'] < 20).any()


def custom_filter_rule_slowkd1(code: str, name: str, kline_df: pd.DataFrame) -> bool:
    """
    规则2：前1个月内出现过SlowKD K值低于20
    SlowKD参数：9,3
    """
    if kline_df.empty or len(kline_df) < 20:
        return False

    df = calculate_slowkd(kline_df, n=9, m=3)
    last_month_df = get_last_month_data(df)

    if last_month_df.empty:
        return False

    return (last_month_df['slowkd_k'] < 20).any()


def custom_filter_rule_kdj2(code: str, name: str, kline_df: pd.DataFrame) -> bool:
    """
    规则3：前1个月KDJ K值先升破80，后跌破50
    KDJ参数：9,3,3
    """
    if kline_df.empty or len(kline_df) < 20:
        return False

    df = calculate_kdj(kline_df, n=9, m1=3, m2=3)
    last_month_df = get_last_month_data(df)

    if last_month_df.empty or len(last_month_df) < 5:
        return False

    last_month_df = last_month_df.sort_values('time_key').reset_index(drop=True)

    # 检测「先升破80 → 后跌破50」的时序变化
    crossed_80 = False
    for _, row in last_month_df.iterrows():
        k_value = row['kdj_k']
        if pd.isna(k_value):
            continue
        if not crossed_80 and k_value > 80:
            crossed_80 = True
        elif crossed_80 and k_value < 50:
            return True
    return False


def custom_filter_rule_slowkd2(code: str, name: str, kline_df: pd.DataFrame) -> bool:
    """
    规则4：前1个月SlowKD K值先升破80，后跌破50
    SlowKD参数：9,3
    """
    if kline_df.empty or len(kline_df) < 20:
        return False

    df = calculate_slowkd(kline_df, n=9, m=3)
    last_month_df = get_last_month_data(df)

    if last_month_df.empty or len(last_month_df) < 5:
        return False

    last_month_df = last_month_df.sort_values('time_key').reset_index(drop=True)

    # 检测「先升破80 → 后跌破50」的时序变化
    crossed_80 = False
    for _, row in last_month_df.iterrows():
        k_value = row['slowkd_k']
        if pd.isna(k_value):
            continue
        if not crossed_80 and k_value > 80:
            crossed_80 = True
        elif crossed_80 and k_value < 50:
            return True
    return False


# ==================== 整合筛选函数（核心） ====================
def integrated_filter_rule(code: str, name: str, kline_df: pd.DataFrame) -> tuple:
    """
    整合8个筛选规则，只读取一次K线数据，同时完成所有条件检测

    :return: (是否满足任一条件: bool, 输出字典)
    输出字典包含：标的代码、名称、最新收盘价、满足的规则列表、满足规则数量
    """
    if kline_df.empty or len(kline_df) < 20:
        return False, None

    # 预先计算KDJ和SlowKD（避免重复计算）
    df_sorted = kline_df.sort_values('time_key').reset_index(drop=True)
    df_kdj = calculate_kdj(df_sorted.copy(), n=9, m1=3, m2=3)
    df_slowkd = calculate_slowkd(df_sorted.copy(), n=9, m=3)

    # 预先计算RSI(6)和RSI(7)（新增）
    df_rsi6 = calculate_rsi(df_sorted.copy(), period=6)
    df_rsi7 = calculate_rsi(df_sorted.copy(), period=7)

    # 获取前1个月数据
    last_month_kdj = get_last_month_data(df_kdj)
    last_month_slowkd = get_last_month_data(df_slowkd)
    last_month_rsi6 = get_last_month_data(df_rsi6)
    last_month_rsi7 = get_last_month_data(df_rsi7)

    matched_rules = []

    # 规则1：KDJ K值低于20
    if not last_month_kdj.empty and (last_month_kdj['kdj_k'] < 20).any():
        matched_rules.append("KDJ_K<20")

    # 规则2：SlowKD K值低于20
    if not last_month_slowkd.empty and (last_month_slowkd['slowkd_k'] < 20).any():
        matched_rules.append("SlowKD_K<20")

    # 规则3：KDJ K值先升破80后跌破50
    if not last_month_kdj.empty and len(last_month_kdj) >= 5:
        crossed_80 = False
        for _, row in last_month_kdj.iterrows():
            k_value = row['kdj_k']
            if pd.notna(k_value):
                if not crossed_80 and k_value > 80:
                    crossed_80 = True
                elif crossed_80 and k_value < 50:
                    matched_rules.append("KDJ_K>80→<50")
                    break

    # 规则4：SlowKD K值先升破80后跌破50
    if not last_month_slowkd.empty and len(last_month_slowkd) >= 5:
        crossed_80 = False
        for _, row in last_month_slowkd.iterrows():
            k_value = row['slowkd_k']
            if pd.notna(k_value):
                if not crossed_80 and k_value > 80:
                    crossed_80 = True
                elif crossed_80 and k_value < 50:
                    matched_rules.append("SlowKD_K>80→<50")
                    break

    # ==================== 新增RSI规则（规则5-8） ====================
    # 规则5：RSI(6) >= 80
    if not last_month_rsi6.empty and (last_month_rsi6['rsi_6'] >= 80).any():
        matched_rules.append("RSI(6)>=80")

    # 规则6：RSI(6) <= 20
    if not last_month_rsi6.empty and (last_month_rsi6['rsi_6'] <= 20).any():
        matched_rules.append("RSI(6)<=20")

    # 规则7：RSI(7) >= 80
    if not last_month_rsi7.empty and (last_month_rsi7['rsi_7'] >= 80).any():
        matched_rules.append("RSI(7)>=80")

    # 规则8：RSI(7) <= 20
    if not last_month_rsi7.empty and (last_month_rsi7['rsi_7'] <= 20).any():
        matched_rules.append("RSI(7)<=20")

    # 返回结果
    if matched_rules:
        latest_close = df_sorted.iloc[-1]['close'] if not df_sorted.empty else None
        latest_kdj_k = df_kdj.iloc[-1]['kdj_k'] if not df_kdj.empty else None
        latest_slowkd_k = df_slowkd.iloc[-1]['slowkd_k'] if not df_slowkd.empty else None
        latest_rsi6 = df_rsi6.iloc[-1]['rsi_6'] if not df_rsi6.empty else None
        latest_rsi7 = df_rsi7.iloc[-1]['rsi_7'] if not df_rsi7.empty else None

        output = {
            'code': code,
            'name': name,
            'latest_close': round(latest_close, 3) if latest_close else None,
            'latest_kdj_k': round(latest_kdj_k, 2) if latest_kdj_k else None,
            'latest_slowkd_k': round(latest_slowkd_k, 2) if latest_slowkd_k else None,
            'latest_rsi6': round(latest_rsi6, 2) if latest_rsi6 else None,
            'latest_rsi7': round(latest_rsi7, 2) if latest_rsi7 else None,
            'matched_rules': ', '.join(matched_rules),
            'matched_count': len(matched_rules)
        }
        return True, output
    else:
        return False, None


# ==================== 主要功能函数 ====================
class FutuStockScreener:
    def __init__(self):
        self.quote_ctx = None
        self.stock_name_cache = {}  # 缓存股票名称

    def connect(self):
        """连接到 OpenD"""
        print(f"正在连接 OpenD ({CONFIG['host']}:{CONFIG['port']})...")
        self.quote_ctx = OpenQuoteContext(host=CONFIG['host'], port=CONFIG['port'])
        print("连接成功！")

    def disconnect(self):
        """断开连接"""
        if self.quote_ctx:
            self.quote_ctx.close()
            print("已断开连接")

    def get_watchlist_stocks(self) -> list:
        """获取自选股列表 (list_part1)"""
        print(f"\n正在获取自选股分组 [{CONFIG['watchlist_group']}] 中的股票...")
        ret, data = self.quote_ctx.get_user_security(CONFIG['watchlist_group'])
        if ret == RET_OK:
            stock_list = data['code'].tolist()
            # 从自选股列表中缓存股票名称
            for _, row in data.iterrows():
                self.stock_name_cache[row['code']] = row['name']
            print(f"获取到 {len(stock_list)} 只自选股")
            return stock_list
        else:
            print(f"获取自选股失败: {data}")
            return []

    def get_all_watchlist_groups(self) -> list:
        """获取所有自选股分组名称"""
        ret, data = self.quote_ctx.get_user_security_group(UserSecurityGroupType.ALL)
        if ret == RET_OK:
            return data['group_name'].tolist()
        return []

    def merge_stock_lists(self, list1: list, list2: list) -> list:
        """合并两个股票列表并去重"""
        merged = list(set(list1 + list2))
        print(f"\n合并后共 {len(merged)} 只股票（去重）")
        return merged

    def batch_get_stock_names(self, stock_list: list):
        """
        批量获取股票名称并缓存
        使用 get_market_snapshot 获取股票基本信息
        """
        # 筛选出还没有名称的股票
        codes_without_name = [code for code in stock_list if code not in self.stock_name_cache]

        if not codes_without_name:
            return

        print(f"\n正在获取 {len(codes_without_name)} 只股票的名称...")

        # 分批获取（每批最多200只）
        batch_size = 200
        for i in range(0, len(codes_without_name), batch_size):
            batch = codes_without_name[i:i + batch_size]
            ret, data = self.quote_ctx.get_market_snapshot(batch)
            if ret == RET_OK:
                for _, row in data.iterrows():
                    self.stock_name_cache[row['code']] = row['name']
            else:
                print(f"  获取股票名称失败: {data}")

            # 避免频率限制
            if i + batch_size < len(codes_without_name):
                time.sleep(0.5)

        print(f"  已缓存 {len(self.stock_name_cache)} 只股票的名称")

    def get_stock_name(self, code: str) -> str:
        """获取单只股票的名称（从缓存中）"""
        return self.stock_name_cache.get(code, '')

    def get_kline_data(self, code: str) -> tuple:
        """获取单只股票的K线数据"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=CONFIG['kline_days'])).strftime('%Y-%m-%d')

        ret, data, page_req_key = self.quote_ctx.request_history_kline(
            code=code,
            start=start_date,
            end=end_date,
            ktype=KLType.K_DAY,
            autype=AuType.QFQ,
            fields=[KL_FIELD.DATE_TIME, KL_FIELD.OPEN, KL_FIELD.CLOSE,
                    KL_FIELD.HIGH, KL_FIELD.LOW],
            max_count=1000
        )

        if ret == RET_OK:
            all_data = data
            while page_req_key is not None:
                ret, data, page_req_key = self.quote_ctx.request_history_kline(
                    code=code,
                    start=start_date,
                    end=end_date,
                    ktype=KLType.K_DAY,
                    autype=AuType.QFQ,
                    fields=[KL_FIELD.DATE_TIME, KL_FIELD.OPEN, KL_FIELD.CLOSE,
                            KL_FIELD.HIGH, KL_FIELD.LOW],
                    max_count=1000,
                    page_req_key=page_req_key
                )
                if ret == RET_OK:
                    all_data = pd.concat([all_data, data], ignore_index=True)

            # 从缓存中获取股票名称
            name = self.get_stock_name(code)
            return True, name, all_data
        else:
            return False, '', data

    def batch_fetch_kline(self, stock_list: list) -> dict:
        """分批获取所有股票的K线数据"""
        print(f"\n开始分批获取K线数据（每批 {CONFIG['batch_size']} 只，间隔 {CONFIG['batch_interval']} 秒）...")

        results = {}
        total = len(stock_list)

        for i in range(0, total, CONFIG['batch_size']):
            batch = stock_list[i:i + CONFIG['batch_size']]
            batch_num = i // CONFIG['batch_size'] + 1
            total_batches = (total + CONFIG['batch_size'] - 1) // CONFIG['batch_size']

            print(f"\n第 {batch_num}/{total_batches} 批:")

            for code in batch:
                success, name, data = self.get_kline_data(code)
                if success:
                    results[code] = {'name': name, 'kline': data}
                    print(f"  ✓ {code} ({name}) - {len(data)} 条记录")
                else:
                    print(f"  ✗ {code} - 获取失败: {data}")

            if i + CONFIG['batch_size'] < total:
                time.sleep(CONFIG['batch_interval'])

        print(f"\n成功获取 {len(results)}/{total} 只股票的K线数据")
        return results

    def screen_stocks(self, kline_data: dict, filter_func) -> pd.DataFrame:
        """根据自定义规则筛选股票"""
        print("\n开始筛选股票（KDJ/SlowKD 4个条件 + RSI 4个条件）...")

        passed_stocks = []

        for code, data in kline_data.items():
            name = data['name']
            kline_df = data['kline']

            try:
                is_passed, output = filter_func(code, name, kline_df)
                if is_passed:
                    passed_stocks.append(output)
                    print(f"  ✓ 通过筛选: {code} ({name}) - 满足条件: {output['matched_rules']}")
            except Exception as e:
                print(f"  ✗ 筛选出错 {code}: {e}")

        if passed_stocks:
            result_df = pd.DataFrame(passed_stocks)
            # 按满足条件数量排序
            result_df = result_df.sort_values('matched_count', ascending=False).reset_index(drop=True)
            return result_df
        else:
            return pd.DataFrame()

    def run(self, custom_list: list = None, filter_func=None):
        """运行主流程"""
        if custom_list is None:
            custom_list = []
        if filter_func is None:
            filter_func = integrated_filter_rule

        try:
            self.connect()

            groups = self.get_all_watchlist_groups()
            print(f"\n可用的自选股分组: {groups}")

            list_part1 = self.get_watchlist_stocks()
            merged_list = self.merge_stock_lists(list_part1, custom_list)

            if not merged_list:
                print("没有股票需要处理！")
                return None

            # 批量获取股票名称（修复：在获取K线之前先获取名称）
            self.batch_get_stock_names(merged_list)

            kline_data = self.batch_fetch_kline(merged_list)

            if not kline_data:
                print("没有获取到有效的K线数据！")
                return None

            result = self.screen_stocks(kline_data, filter_func)

            print("\n" + "=" * 100)
            print("筛选结果（KDJ 9,3,3 / SlowKD 9,3 / RSI 6,7）")
            print("=" * 100)
            print("规则说明：")
            print("  KDJ_K<20       : 前1个月内KDJ K值曾低于20")
            print("  SlowKD_K<20    : 前1个月内SlowKD K值曾低于20")
            print("  KDJ_K>80→<50   : 前1个月内KDJ K值先升破80后跌破50")
            print("  SlowKD_K>80→<50: 前1个月内SlowKD K值先升破80后跌破50")
            print("  RSI(6)>=80     : 前1个月内RSI(6)曾达到80或以上（超买）")
            print("  RSI(6)<=20     : 前1个月内RSI(6)曾达到20或以下（超卖）")
            print("  RSI(7)>=80     : 前1个月内RSI(7)曾达到80或以上（超买）")
            print("  RSI(7)<=20     : 前1个月内RSI(7)曾达到20或以下（超卖）")
            print("=" * 100)

            if result.empty:
                print("\n没有股票满足任一筛选条件")
            else:
                print(f"\n共 {len(result)} 只股票满足条件:\n")
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_colwidth', 50)
                print(result.to_string(index=False))

            return result

        finally:
            self.disconnect()


# ==================== 主程序入口 ====================
if __name__ == '__main__':
    screener = FutuStockScreener()

    result = screener.run(
        custom_list=list_part2,
        filter_func=integrated_filter_rule
    )

    if result is not None and not result.empty:
        filename = f"kdj_slowkd_rsi_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        result.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {filename}")
