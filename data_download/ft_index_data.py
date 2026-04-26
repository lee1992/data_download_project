

FUTU_OPEND_HOST = "127.0.0.1"
FUTU_OPEND_PORT = 11111
# -*- coding: utf-8 -*-
import pandas as pd
import time
import os
from datetime import datetime
from futu import *

quote_ctx = OpenQuoteContext(host=FUTU_OPEND_HOST, port=FUTU_OPEND_PORT)
quote_ctx.set_handler(None)  # 关闭推送，仅查询历史数据
ret, plate_data = quote_ctx.get_plate_list(Market.US,Plate.ALL)

plate_data['']
quote_ctx.get_plate_stock(plate_code, sort_field=SortField.CODE, ascend=True)