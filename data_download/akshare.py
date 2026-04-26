import akshare as ak
import pandas as pd

import akshare as ak

# 美股日线数据（无需VPN）
#us_list2= ak.get_us_stock_name()
# 获取所有美股实时列表，从中找 NVDA 的正确 symbol
#us_spot = ak.stock_us_spot_em()
#stock_us_famous_spot_em_df = ak.stock_us_famous_spot_em()(symbol='科技类')
df_us = ak.stock_us_daily(symbol="QQQ", adjust="qfq")
stock_us_hist_min_em_df = ak.stock_us_hist_min_em(symbol="NVDA",start_date='2015-01-01',end_date='2026-04-25')
# A股分钟线数据
df_cn = ak.stock_zh_a_minute(symbol="600519", period="5min")
# 直接返回pandas DataFrame，和你现有回测代码完全兼容