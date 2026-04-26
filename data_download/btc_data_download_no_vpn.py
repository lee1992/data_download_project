# -*- coding: utf-8 -*-
"""
=============================================================================
 方案2：无需代理！通过 CryptoCompare 免费API 下载BTC历史K线数据
 适用场景：国内用户，无代理，无VPN
=============================================================================

 【数据来源】CryptoCompare（https://www.cryptocompare.com）
   - 国内可以直接访问，无需代理
   - 免费API，无需注册即可使用（有IP限流，建议注册获取免费API Key）
   - 支持分钟/小时/日级别数据
   - 数据来源于多个交易所的聚合价格（CCCAGG）

 【获取免费 API Key（推荐，可选）】
   1. 打开 https://www.cryptocompare.com
   2. 注册账号（免费）
   3. 登录后访问 https://www.cryptocompare.com/cryptopian/api-keys
   4. 创建一个 API Key，复制到下方 API_KEY 配置中
   5. 免费版限制：每月 100,000 次调用，完全够用

 【数据说明】
   - 分钟数据：最多获取最近7天的数据（API限制）
   - 小时数据：无明显时间限制，可获取多年历史
   - 日线数据：无明显时间限制，可获取多年历史

 【注意事项】
   - 免费版无API Key时，IP限流较严格（约每秒20次）
   - 有API Key时限流宽松很多
   - 数据为UTC时间，脚本会自动转换为北京时间（可选）
=============================================================================
"""

import pandas as pd
import time
import os
import requests
from datetime import datetime, timedelta

# -------------------------- 【用户配置区】 --------------------------

# 1. 交易币种和计价币种
COIN = "BTC"          # 加密货币符号
CURRENCY = "USDT"     # 计价货币（USDT, USD, CNY 等）

# 2. 数据时间范围
START_DATE = "2024-04-02"
END_DATE = "2026-04-02"

# 3. K线周期
# 可选：'1m'(1分钟), '5m', '15m', '30m', '1h'(小时), '1d'(日线)
# ⚠️ 注意：分钟级别数据最多只能获取最近7天！
KLINE_TIMEFRAME = "1h"

# 4. 保存路径
SAVE_PATH = "./btc_history_data/"

# 5. API Key（可选，留空也能用，但限流更严格）
#    获取方式：https://www.cryptocompare.com/cryptopian/api-keys
API_KEY = ""  # ← 填入你的免费API Key，如："xxxxxxxxxxxxxxxxxxxxxxxx"

# 6. 是否转换为北京时间
CONVERT_TO_BEIJING_TIME = True

# 7. 超时和重试配置
TIMEOUT = 30          # 请求超时（秒）
MAX_RETRIES = 5       # 最大重试次数
# -------------------------------------------------------------------


# CryptoCompare API 基础URL
BASE_URL = "https://min-api.cryptocompare.com/data/v2"

# 时间周期映射
TIMEFRAME_CONFIG = {
    '1m':  {'endpoint': 'histominute', 'aggregate': 1,  'name': '1分钟', 'max_limit': 2000},
    '5m':  {'endpoint': 'histominute', 'aggregate': 5,  'name': '5分钟', 'max_limit': 2000},
    '15m': {'endpoint': 'histominute', 'aggregate': 15, 'name': '15分钟', 'max_limit': 2000},
    '30m': {'endpoint': 'histominute', 'aggregate': 30, 'name': '30分钟', 'max_limit': 2000},
    '1h':  {'endpoint': 'histohour',   'aggregate': 1,  'name': '1小时', 'max_limit': 2000},
    '2h':  {'endpoint': 'histohour',   'aggregate': 2,  'name': '2小时', 'max_limit': 2000},
    '4h':  {'endpoint': 'histohour',   'aggregate': 4,  'name': '4小时', 'max_limit': 2000},
    '1d':  {'endpoint': 'histoday',    'aggregate': 1,  'name': '日线',  'max_limit': 2000},
}


def test_connection():
    """测试是否能连接 CryptoCompare API"""
    print("🔍 正在测试 CryptoCompare API 连接...")
    try:
        headers = {}
        if API_KEY:
            headers['authorization'] = f'Apikey {API_KEY}'

        resp = requests.get(
            f"{BASE_URL}/histoday",
            params={'fsym': 'BTC', 'tsym': 'USD', 'limit': 1},
            headers=headers,
            timeout=TIMEOUT
        )
        data = resp.json()

        if data.get('Response') == 'Success':
            print("✅ 连接成功！CryptoCompare API 可用\n")
            return True
        else:
            print(f"❌ API返回错误：{data.get('Message', '未知错误')}")
            return False
    except requests.exceptions.Timeout:
        print("❌ 连接超时，请检查网络")
        return False
    except Exception as e:
        print(f"❌ 连接失败：{str(e)[:150]}")
        return False


def fetch_batch(endpoint, coin, currency, aggregate, limit, to_ts=None):
    """
    单次API请求，获取一批K线数据

    Args:
        endpoint: 'histominute' / 'histohour' / 'histoday'
        coin: 加密货币符号
        currency: 计价货币
        aggregate: 聚合周期
        limit: 返回数据量（最大2000）
        to_ts: 截止时间戳（秒），None则为当前时间

    Returns:
        (data_list, earliest_timestamp, has_more)
    """
    url = f"{BASE_URL}/{endpoint}"
    params = {
        'fsym': coin,
        'tsym': currency,
        'limit': limit,
        'aggregate': aggregate,
    }
    if to_ts:
        params['toTs'] = to_ts

    headers = {}
    if API_KEY:
        headers['authorization'] = f'Apikey {API_KEY}'

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
            data = resp.json()

            if data.get('Response') == 'Success':
                records = data['Data']['Data']
                # 过滤掉成交量为0的无效数据
                valid_records = [r for r in records if r.get('volumeto', 0) > 0 or r.get('close', 0) > 0]
                if valid_records:
                    earliest_ts = valid_records[0]['time']
                    return valid_records, earliest_ts, True
                else:
                    return [], None, False

            elif data.get('Response') == 'Error':
                msg = data.get('Message', '')
                if 'rate limit' in msg.lower():
                    wait = 10
                    print(f"   🚫 API限流，等待{wait}秒...")
                    time.sleep(wait)
                    continue
                else:
                    print(f"   ❌ API错误：{msg}")
                    return [], None, False

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"   ⏱️ 超时，{wait}秒后重试（{attempt+1}/{MAX_RETRIES}）")
                time.sleep(wait)
            else:
                print(f"   ❌ 多次超时")
                return [], None, False
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 3
                print(f"   ⚠️ 异常：{str(e)[:80]}，{wait}秒后重试")
                time.sleep(wait)
            else:
                return [], None, False

    return [], None, False


def fetch_btc_kline_data(
        coin: str,
        currency: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        save_path: str
):
    """
    下载BTC历史K线数据（通过 CryptoCompare 免费API，无需代理）
    采用从后往前的分页方式获取数据
    """

    # 检查周期配置
    if timeframe not in TIMEFRAME_CONFIG:
        print(f"❌ 不支持的K线周期：{timeframe}")
        print(f"   支持的周期：{', '.join(TIMEFRAME_CONFIG.keys())}")
        return None

    config = TIMEFRAME_CONFIG[timeframe]

    # 分钟级别数据的时间限制提醒
    if config['endpoint'] == 'histominute':
        days_span = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days
        if days_span > 7:
            print(f"⚠️ 注意：CryptoCompare 分钟级别数据最多支持最近7天！")
            print(f"   你设置的时间跨度为 {days_span} 天")
            print(f"   系统将自动调整开始时间为最近7天")
            adjusted_start = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
            start_date = adjusted_start
            print(f"   调整后范围：{start_date} → {end_date}\n")

    # 测试连接
    if not test_connection():
        return None

    # 时间转换
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    # 不超过当前时间
    now_ts = int(datetime.now().timestamp())
    end_ts = min(end_ts, now_ts)

    print(f"{'='*70}")
    print(f"📥 开始下载：{coin}/{currency} | 周期：{config['name']}")
    print(f"   时间范围：{start_date} → {end_date}")
    print(f"   数据来源：CryptoCompare（免费，无需代理）")
    if API_KEY:
        print(f"   API Key：已配置 ✅")
    else:
        print(f"   API Key：未配置（免费使用，限流较严格）")
    print(f"{'='*70}\n")

    all_records = []
    current_to_ts = end_ts
    request_count = 0
    limit_per_request = config['max_limit']

    # 从后往前获取数据（CryptoCompare 的 toTs 参数）
    while current_to_ts > start_ts:
        records, earliest_ts, has_more = fetch_batch(
            endpoint=config['endpoint'],
            coin=coin,
            currency=currency,
            aggregate=config['aggregate'],
            limit=limit_per_request,
            to_ts=current_to_ts
        )

        request_count += 1

        if not records:
            print(f"\n✅ 无更多数据，获取完毕！")
            break

        all_records = records + all_records  # 前插（保持时间顺序）

        # 获取这批数据中最早的时间
        batch_earliest = records[0]['time']
        batch_latest = records[-1]['time']
        batch_earliest_str = datetime.fromtimestamp(batch_earliest).strftime('%Y-%m-%d %H:%M')
        batch_latest_str = datetime.fromtimestamp(batch_latest).strftime('%Y-%m-%d %H:%M')

        print(f"   ✓ 批次 {request_count:3d} | +{len(records):4d}条 | 范围：{batch_earliest_str} → {batch_latest_str}")

        # 如果已经到达或超过开始时间
        if batch_earliest <= start_ts:
            print(f"\n✅ 已覆盖目标时间范围！")
            break

        # 下一次请求的截止时间 = 当前最早时间 - 1
        current_to_ts = batch_earliest - 1

        # 防限流
        if API_KEY:
            time.sleep(0.2)  # 有Key限流宽松
        else:
            time.sleep(0.5)  # 无Key限流严格，慢一点

        if request_count % 20 == 0:
            time.sleep(2)  # 额外休息

    if not all_records:
        print(f"⚠️ 未获取到任何数据")
        return None

    # 转为DataFrame
    df = pd.DataFrame(all_records)

    # 重命名列
    df = df.rename(columns={
        'time': '时间戳',
        'open': '开盘价',
        'high': '最高价',
        'low': '最低价',
        'close': '收盘价',
        'volumefrom': '成交量(BTC)',
        'volumeto': '成交额(USDT)',
    })

    # 时间转换
    df['时间(UTC)'] = pd.to_datetime(df['时间戳'], unit='s')

    if CONVERT_TO_BEIJING_TIME:
        df['时间'] = df['时间(UTC)'] + timedelta(hours=8)
    else:
        df['时间'] = df['时间(UTC)']

    # 选取需要的列
    keep_cols = ['时间', '开盘价', '最高价', '最低价', '收盘价', '成交量(BTC)', '成交额(USDT)']
    df = df[keep_cols]

    # 过滤时间范围
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if CONVERT_TO_BEIJING_TIME:
        start_dt = start_dt + timedelta(hours=8)
        end_dt = end_dt + timedelta(hours=8)

    df = df[(df['时间'] >= start_dt) & (df['时间'] <= end_dt)]

    # 去重排序
    df = df.drop_duplicates(subset=['时间'], keep='first').sort_values('时间').reset_index(drop=True)

    # 过滤无效数据（开盘价为0的行）
    df = df[df['开盘价'] > 0].reset_index(drop=True)

    print(f"\n{'='*70}")
    print(f"✅ 下载完成！")
    print(f"   共获取：{len(df)} 条 {config['name']} 数据")
    if len(df) > 0:
        print(f"   时间范围：{df['时间'].min()} → {df['时间'].max()}")
    print(f"   请求次数：{request_count} 次")
    print(f"{'='*70}")

    # 保存
    os.makedirs(save_path, exist_ok=True)
    filename = f"{save_path}{coin}_{currency}_{timeframe}_{start_date}_{end_date}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"💾 已保存：{filename}")

    return df


if __name__ == "__main__":
    df = fetch_btc_kline_data(
        coin=COIN,
        currency=CURRENCY,
        timeframe=KLINE_TIMEFRAME,
        start_date=START_DATE,
        end_date=END_DATE,
        save_path=SAVE_PATH
    )

    if df is not None:
        print(f"\n📊 数据预览（前5行）：")
        print(df.head(5).to_string(index=False))
        print(f"\n📊 数据预览（后5行）：")
        print(df.tail(5).to_string(index=False))

        print(f"\n📈 价格统计：")
        print(f"   最高价：{df['最高价'].max():.2f}")
        print(f"   最低价：{df['最低价'].min():.2f}")
        print(f"   均价  ：{df['收盘价'].mean():.2f}")
