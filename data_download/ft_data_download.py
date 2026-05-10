# -*- coding: utf-8 -*-
import pandas as pd
import time
import os
from datetime import datetime
from futu import *

# -------------------------- 【用户配置区】所有参数都在这里修改 --------------------------
# 1. 你要下载的股票/期货列表（富途代码格式）
STOCK_LIST = [
    "HK.HSImain",  # 恒指主连
     "SH.510500",  # 恒指主连
     "SH.000001",  # 恒指主连
     "US.AMD",
      "US.NVDA",
     'SZ.002466',
     "SZ.300014",
     "SZ.002460",
     "US.QQQ",
    # "US.NQmain",  # 纳斯达克100主连（需要美股期货权限）  需要额外买
]

# 2. 数据时间范围
START_DATE = "2017-04-22"
END_DATE = "2026-04-29"

# 3. K线类型配置（核心修改点）
# 可选值：KLType.K_1M, KLType.K_5M, KLType.K_15M, KLType.K_30M, KLType.K_60M, KLType.K_DAY等
KLINE_TYPE = KLType.K_30M#KLType.K_60M  # 这里改成你要的周期
KLINE_NAME = KLINE_TYPE# 用于打印和文件名的显示名称

# 4. 数据保存路径（自动根据K线类型创建子文件夹）
BASE_SAVE_PATH = "./futu_data/"

# 5. 富途OpenD连接配置
FUTU_OPEND_HOST = "127.0.0.1"
FUTU_OPEND_PORT = 11111


# ----------------------------------------------------------------------------------------

def download_kline_data(
        stock_code: str,
        kline_type: KLType,
        kline_name: str,
        start_date: str,
        end_date: str,
        save_path: str,
        quote_ctx: OpenQuoteContext
):
    """
    下载单只标的的K线数据（支持分页获取+频率限制处理）
    """
    print(f"正在下载：{stock_code} | 时间范围：{start_date} - {end_date} | {kline_name}K线")

    all_data = []
    page_req_key = None  # 分页密钥，第一次为空
    request_count = 0  # 请求计数，用于频率控制

    while True:
        try:
            # 调用富途API：获取历史K线（支持分页）
            ret_code, data, next_page_req_key = quote_ctx.request_history_kline(
                code=stock_code,
                start=start_date,
                end=end_date,
                ktype=kline_type,
                autype=AuType.NONE,
                page_req_key=page_req_key  # 传入分页密钥
            )

            request_count += 1

            # 处理频率限制（核心修改点2）
            if ret_code == RET_ERROR:
                if "频率太高" in str(data):
                    print(f"⚠️ 触发频率限制，休息30秒后继续... (已请求{request_count}次)")
                    time.sleep(30)
                    continue  # 休息后重试
                else:
                    print(f"❌ {stock_code} 下载失败：{data}")
                    return None

            # 数据为空，说明已经获取完毕
            if data.empty:
                print(f"✅ {stock_code} 数据获取完毕，无更多数据")
                break

            # 追加数据到总列表
            all_data.append(data)
            print(f"   已获取 {len(data)} 条数据，累计 {len(pd.concat(all_data))} 条")

            # 更新分页密钥
            page_req_key = next_page_req_key

            # 如果没有下一页密钥，说明获取完毕
            if not page_req_key:
                print(f"✅ {stock_code} 所有分页获取完毕")
                break

            # 简单的频率控制：每请求50次休息1秒（避免触发限制）
            if request_count % 50 == 0:
                time.sleep(1)

        except Exception as e:
            print(f"❌ {stock_code} 发生异常：{str(e)}")
            time.sleep(5)
            continue

    # 合并所有数据
    if not all_data:
        print(f"⚠️ {stock_code} 无数据")
        return None

    final_df = pd.concat(all_data, ignore_index=True)

    # 数据清洗：格式化时间
    final_df["time_key"] = pd.to_datetime(final_df["time_key"])

    # 去重（防止分页重复）
    final_df = final_df.drop_duplicates(subset=["time_key"], keep="first").sort_values("time_key").reset_index(
        drop=True)

    print(f"✅ {stock_code} 下载完成，共 {len(final_df)} 条{kline_name}数据")
    return final_df


def batch_download_kline_data():
    """批量下载股票列表的K线数据"""
    # 根据K线类型创建保存文件夹
    full_save_path = os.path.join(BASE_SAVE_PATH, f"{KLINE_NAME}/")
    os.makedirs(full_save_path, exist_ok=True)

    # 初始化富途行情连接
    quote_ctx = OpenQuoteContext(host=FUTU_OPEND_HOST, port=FUTU_OPEND_PORT)
    quote_ctx.set_handler(None)  # 关闭推送，仅查询历史数据

    try:
        # 循环下载每只股票
        for code in STOCK_LIST:
            df = download_kline_data(
                stock_code=code,
                kline_type=KLINE_TYPE,
                kline_name=KLINE_NAME,
                start_date=START_DATE,
                end_date=END_DATE ,
                save_path=full_save_path ,
                quote_ctx=quote_ctx
            )

            if df is not None:
                # 保存为CSV文件（文件名包含K线类型）
                save_file = f"{full_save_path}{code.replace('.', '_')}_{KLINE_NAME}_{START_DATE}_{END_DATE}.csv"
                df.to_csv(save_file, index=False, encoding="utf-8-sig")
                print(f"   数据已保存至：{save_file}\n")

    finally:
        # 关闭连接（必须执行）
        quote_ctx.close()
        print("\n🎉 所有任务执行完毕，数据保存在：", full_save_path)


if __name__ == "__main__":
    batch_download_kline_data()
    #ret, idx_df = quote_ctx.get_stock_basicinfo(Market.COMEX, SecurityType.IDX)