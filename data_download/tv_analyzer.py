"""
TradingView 自动截图 + 多周期合图 + 豆包/Sider 投喂系统
==========================================================
依赖安装：
    pip install playwright pillow pyyaml
    playwright install chromium

目录结构（自动创建）：
    charts/
        2026-04-25/
            NASDAQ_MU/
                NASDAQ_MU_12M.png
                NASDAQ_MU_1M.png
                ...
                NASDAQ_MU_combined.png   ← 合图，直接投喂
    config.yaml
    results/
        2026-04-25_MU_analysis.txt
"""

import asyncio
import os
import re
import yaml
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ─── 默认配置（如果 config.yaml 不存在则写入）─────────────────────────────────

DEFAULT_CONFIG = {
    "tradingview": {
        "base_url": "https://www.tradingview.com/chart/",
        "chart_id": "",          # 留空则用默认图表；填入你的图表ID可固定模板
        "theme": "dark",
        "width": 1920,
        "height": 1080,
        "chart_area_top_offset": 60,   # 裁去顶部工具栏的像素
        "wait_after_load": 3000,       # 图表加载等待(ms)
        "wait_after_interval": 2000,   # 切换周期后等待(ms)
        "user_data_dir": "./tv_profile", # 保留登录态的 profile 目录
        "navigation_timeout": 45000,      # 页面打开超时(ms)
        "goto_retries": 3,                # 导航重试次数
        "homepage_check": False,          # 是否先打开首页探测连通性
        "proxy": ""                      # 例如 http://127.0.0.1:7890
    },

    # 周期配置
    # key = 显示名称, value = TradingView 快捷键字符串（对应键盘快捷键触发方式）
    # 也可以用 click_selector 方式，见 INTERVAL_CLICK_MAP
    "intervals": ["12M", "6M", "3M", "1M", "1W", "1D", "240", "60", "30", "15", "5"],

    "symbols": [
        "NASDAQ:NVDA",
        "NASDAQ:AMD",
        "NASDAQ:MU",
        "NASDAQ:MRVL",
        "NASDAQ:PLTR",
        "NASDAQ:META",
        "NASDAQ:AAPL",
        "NASDAQ:MSFT",
        "NASDAQ:AMZN",
        "NASDAQ:TSLA",
        "NYSE:GE",
        "NYSE:CAT",
        "NYSE:JPM",
        "NYSE:LLY",
        "SP:SPX",
        "NASDAQ:NDX",
        "OANDA:XAUUSD",
        "OANDA:XAGUSD",
        "NYMEX:CL1!",
    ],

    # 合图设置
    "merge": {
        "enabled": True,
        "cols": 3,                # 每行放几张
        "thumb_width": 960,       # 合图中每张缩略图宽度
        "label_height": 40,       # 每张图下方标注高度
        "font_size": 24,
        "background": "#1a1a2e",  # 合图背景色
        "max_per_sheet": 7,       # 每张合图最多放几张（对齐豆包/Sider限制）
        "output_suffix": "_combined"
    },

    # AI 分析提示词
    "prompt": {
        "wave_analysis": """请基于我上传的 TradingView 蜡烛图进行波浪理论分析。

图片说明：
- 每张合图包含同一标的的多个周期（从年线到5分钟），从左到右、从上到下依次排列
- 请先从年线/月线建立大级别方向，再向下细化至日线、小时线和分钟线

分析要求：
1. 至少细分四个层级浪型（周期级 / 主浪级 / 中级浪 / 小级浪）
2. 判断当前所处浪型结构（推动浪 / 调整浪，以及具体浪号）
3. 给出 2-3 个高概率结构路径，并说明触发条件和失效条件
4. 结合江恩和道氏原理，标出关键支撑/压力位
5. 判断该标的是否属于：刚启动(3浪初期) / 主升延长 / 5浪延伸 / 调整中 / 见顶嫌疑
6. 最终给出一个综合评分（1-10分）和买卖建议

请严格只依据图片分析，不要使用外部数据。每个路径请单独标注概率（高/中/低）。""",

        "screening": """以下是一组股票的多周期技术图表。请帮我做筛选评估：

对每个标的，请：
1. 标注当前所在浪型结构
2. 评估结构确定性（高/中/低）
3. 给出综合评分（1-10分）
4. 最终分类：A（强烈做多候选）/ B（考虑做多）/ C（观望）/ D（回避）

最后请汇总输出一个表格，列出：标的 | 浪型 | 评分 | 分类"""
    },

    "output": {
        "charts_dir": "./charts",
        "results_dir": "./results",
    }
}


# ─── TradingView 周期快捷键映射 ────────────────────────────────────────────────
# TradingView 通过顶部工具栏的时间周期按钮切换，我们直接点击 UI
# 格式：interval_key -> 在时间周期下拉框中可能显示的文本或特殊标识

INTERVAL_LABELS = {
    "12M": "1Y",    # 年线 (12 months)
    "6M":  "6M",
    "3M":  "3M",
    "1M":  "1M",    # 月线
    "1W":  "1W",    # 周线
    "1D":  "1D",    # 日线
    "240": "4H",    # 4小时
    "60":  "1H",    # 1小时
    "30":  "30",    # 30分钟
    "15":  "15",    # 15分钟
    "5":   "5",     # 5分钟
    "1":   "1",     # 1分钟
}


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def load_config(path="config.yaml") -> dict:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(DEFAULT_CONFIG, f, allow_unicode=True, default_flow_style=False)
        print(f"[CONFIG] 已生成默认配置文件：{path}，请按需修改后重新运行。")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def sanitize_symbol(symbol: str) -> str:
    """NASDAQ:MU -> NASDAQ_MU"""
    return symbol.replace(":", "_")


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def get_proxy_settings(tv_cfg: dict):
    proxy = tv_cfg.get("proxy")
    if not proxy:
        return None
    if isinstance(proxy, str):
        proxy = proxy.strip()
        return {"server": proxy} if proxy else None
    if isinstance(proxy, dict) and proxy.get("server"):
        return proxy
    return None


async def safe_goto(page, url: str, timeout_ms: int = 45000, retries: int = 3, wait_until: str = "domcontentloaded"):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            log(f"    导航到：{url}  (第{attempt}/{retries}次)")
            await page.goto(url, timeout=timeout_ms, wait_until=wait_until)
            return
        except Exception as e:
            last_err = e
            log(f"    [WARN] 打开失败：{e}")
            if attempt < retries:
                await asyncio.sleep(min(2 * attempt, 5))
    raise last_err


# ─── 截图核心（Playwright）────────────────────────────────────────────────────

async def take_screenshots(cfg: dict, date_str: str):
    """
    遍历所有 symbol × interval 进行截图
    返回：{symbol: {interval: filepath}}
    """
    from playwright.async_api import async_playwright

    tv_cfg = cfg["tradingview"]
    charts_root = Path(cfg["output"]["charts_dir"]) / date_str
    symbols = cfg["symbols"]
    intervals = cfg["intervals"]
    results = {}

    async with async_playwright() as p:
        # 使用持久化 context 保留登录态
        launch_kwargs = {
            "user_data_dir": tv_cfg["user_data_dir"],
            "headless": False,  # 改为 True 可无头运行，但首次需 False 手动登录
            "viewport": {"width": tv_cfg["width"], "height": tv_cfg["height"]},
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--start-maximized",
            ]
        }
        proxy_settings = get_proxy_settings(tv_cfg)
        if proxy_settings:
            launch_kwargs["proxy"] = proxy_settings

        context = await p.chromium.launch_persistent_context(**launch_kwargs)

        page = context.pages[0] if context.pages else await context.new_page()
        page.set_default_navigation_timeout(tv_cfg.get("navigation_timeout", 45000))
        page.set_default_timeout(tv_cfg.get("navigation_timeout", 45000))

        # 某些网络环境打开首页更容易超时，默认跳过首页，直接进图表
        if tv_cfg.get("homepage_check", False):
            await safe_goto(
                page,
                tv_cfg.get("base_url", "https://www.tradingview.com/chart/"),
                timeout_ms=tv_cfg.get("navigation_timeout", 45000),
                retries=tv_cfg.get("goto_retries", 3),
            )
            await asyncio.sleep(2)

        current_symbol = None

        for symbol in symbols:
            sym_safe = sanitize_symbol(symbol)
            sym_dir = charts_root / sym_safe
            ensure_dirs(str(sym_dir))
            results[symbol] = {}

            # 构建图表URL
            chart_id = tv_cfg.get("chart_id", "")
            if chart_id:
                url = f"https://www.tradingview.com/chart/{chart_id}/?symbol={symbol}"
            else:
                url = f"https://www.tradingview.com/chart/?symbol={symbol}"

            # 若切换标的则导航
            if current_symbol != symbol:
                log(f"  → 打开标的：{symbol}")
                await safe_goto(
                    page,
                    url,
                    timeout_ms=tv_cfg.get("navigation_timeout", 45000),
                    retries=tv_cfg.get("goto_retries", 3),
                )
                await asyncio.sleep(tv_cfg["wait_after_load"] / 1000)
                # 关闭可能弹出的欢迎/提示弹窗
                await dismiss_popups(page)
                current_symbol = symbol

            for interval in intervals:
                label = INTERVAL_LABELS.get(interval, interval)
                out_path = str(sym_dir / f"{sym_safe}_{label}.png")

                log(f"    截图：{symbol} @ {label}")
                success = await set_interval_and_screenshot(
                    page, interval, label, out_path, tv_cfg
                )
                if success:
                    results[symbol][interval] = out_path
                else:
                    log(f"    [WARN] 切换周期失败：{interval}")

            log(f"  ✓ {symbol} 截图完成（{len(results[symbol])}/{len(intervals)} 个周期）")

        await context.close()

    return results


async def dismiss_popups(page):
    """尝试关闭 TradingView 常见弹窗"""
    selectors = [
        "button[aria-label='Close']",
        ".tv-dialog__close",
        ".modal-dialog__close",
        "[data-role='toast-close']",
        ".js-dialog-close",
    ]
    for sel in selectors:
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=800):
                await btn.click()
                await asyncio.sleep(0.5)
        except Exception:
            pass


async def set_interval_and_screenshot(page, interval: str, label: str, out_path: str, tv_cfg: dict) -> bool:
    """
    切换 TradingView 图表周期并截图
    策略：优先用 URL 参数切换（最可靠），备用 UI 点击
    """
    try:
        # ── 方法1：通过 URL interval 参数切换（最稳定）──
        current_url = page.url
        new_url = update_url_interval(current_url, interval)
        if new_url != current_url:
            await safe_goto(
                page,
                new_url,
                timeout_ms=tv_cfg.get("navigation_timeout", 45000),
                retries=tv_cfg.get("goto_retries", 3),
            )
        await asyncio.sleep(tv_cfg["wait_after_interval"] / 1000)

        # ── 方法2：如果URL方式失败，用UI点击 ──
        # await click_interval_button(page, label)

        await dismiss_popups(page)

        # 截图整个页面，裁掉顶部工具栏
        screenshot_bytes = await page.screenshot(full_page=False)
        img = Image.open(
            __import__("io").BytesIO(screenshot_bytes)
        )

        # 裁去顶部导航栏、留核心图表区域
        top = tv_cfg.get("chart_area_top_offset", 60)
        img_cropped = img.crop((0, top, img.width, img.height))
        img_cropped.save(out_path, "PNG", optimize=True)
        return True

    except Exception as e:
        log(f"    [ERROR] 截图失败 {interval}: {e}")
        return False


def update_url_interval(url: str, interval: str) -> str:
    """在 TradingView URL 中更新 interval 参数"""
    import urllib.parse

    # TradingView chart URL 格式示例:
    # https://www.tradingview.com/chart/xxxxx/?symbol=NASDAQ:MU&interval=D
    tv_interval_map = {
        "12M": "12M", "6M": "6M", "3M": "3M",
        "1M":  "M",   "1W": "W",  "1D": "D",
        "240": "240", "60": "60", "30": "30",
        "15":  "15",  "5":  "5",  "1":  "1",
    }
    tv_val = tv_interval_map.get(interval, interval)

    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    params["interval"] = [tv_val]
    new_query = urllib.parse.urlencode({k: v[0] for k, v in params.items()})
    new_url = parsed._replace(query=new_query).geturl()
    return new_url


async def click_interval_button(page, label: str):
    """
    备用方法：点击 TradingView 顶部时间周期按钮
    TradingView 的周期按钮通常是 .chart-toolbar 内的按钮
    """
    selectors = [
        f"[data-value='{label}']",
        f"button:has-text('{label}')",
        f".tv-bar-timeframe__item:has-text('{label}')",
    ]
    for sel in selectors:
        try:
            el = page.locator(sel).first
            if await el.is_visible(timeout=1000):
                await el.click()
                return
        except Exception:
            pass

    # 如果直接找不到，尝试打开下拉
    try:
        more_btn = page.locator("button[aria-label='Time interval']").first
        if await more_btn.is_visible(timeout=1000):
            await more_btn.click()
            await asyncio.sleep(0.5)
            option = page.locator(f"[role='option']:has-text('{label}')").first
            if await option.is_visible(timeout=1000):
                await option.click()
    except Exception:
        pass


# ─── 图片合并 ─────────────────────────────────────────────────────────────────

def merge_images_for_symbol(
    symbol: str,
    interval_paths: dict,
    intervals: list,
    merge_cfg: dict,
    out_dir: str
) -> list:
    """
    将同一标的的多周期截图合并成 1-N 张合图
    返回合图路径列表
    """
    cols = merge_cfg["cols"]
    thumb_w = merge_cfg["thumb_width"]
    label_h = merge_cfg["label_height"]
    font_size = merge_cfg["font_size"]
    bg_color = merge_cfg["background"]
    max_per = merge_cfg["max_per_sheet"]
    suffix = merge_cfg["output_suffix"]

    sym_safe = sanitize_symbol(symbol)

    # 按配置的 intervals 顺序排列（年→分，从大周期到小周期）
    ordered_paths = []
    for iv in intervals:
        label = INTERVAL_LABELS.get(iv, iv)
        if iv in interval_paths:
            ordered_paths.append((label, interval_paths[iv]))

    if not ordered_paths:
        log(f"  [WARN] {symbol} 无可用截图，跳过合图")
        return []

    # 分批（每批 max_per 张）
    batches = [ordered_paths[i:i+max_per] for i in range(0, len(ordered_paths), max_per)]
    output_files = []

    for batch_idx, batch in enumerate(batches):
        rows = (len(batch) + cols - 1) // cols

        # 计算缩略图高度（等比缩放）
        try:
            sample = Image.open(batch[0][1])
            ratio = sample.height / sample.width
            thumb_h = int(thumb_w * ratio)
        except Exception:
            thumb_h = int(thumb_w * 9 / 16)

        canvas_w = cols * thumb_w
        canvas_h = rows * (thumb_h + label_h)

        canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
        draw = ImageDraw.Draw(canvas)

        # 尝试加载字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/Arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

        for idx, (label, img_path) in enumerate(batch):
            row = idx // cols
            col = idx % cols
            x = col * thumb_w
            y = row * (thumb_h + label_h)

            try:
                img = Image.open(img_path)
                img_resized = img.resize((thumb_w, thumb_h), Image.LANCZOS)
                canvas.paste(img_resized, (x, y))
            except Exception as e:
                log(f"  [WARN] 无法读取图片 {img_path}: {e}")
                # 放置占位色块
                placeholder = Image.new("RGB", (thumb_w, thumb_h), "#333344")
                canvas.paste(placeholder, (x, y))

            # 绘制标签
            label_text = f"{sym_safe}  |  {label}"
            label_y = y + thumb_h
            draw.rectangle([x, label_y, x + thumb_w, label_y + label_h], fill="#0d0d1a")
            draw.text(
                (x + 10, label_y + 6),
                label_text,
                fill="#e0e0e0",
                font=font
            )

        # 在合图左上角写标题
        title = f"{symbol}  —  多周期波浪分析图  （{today_str()}）  第{batch_idx+1}张/共{len(batches)}张"
        draw.rectangle([0, 0, canvas_w, label_h], fill="#060614")
        draw.text((10, 8), title, fill="#88ccff", font=font)

        # 保存
        batch_suffix = f"{suffix}_{batch_idx+1}" if len(batches) > 1 else suffix
        out_path = str(Path(out_dir) / f"{sym_safe}{batch_suffix}.png")
        canvas.save(out_path, "PNG", optimize=True, quality=95)
        output_files.append(out_path)
        log(f"  ✓ 合图保存：{out_path}  （{len(batch)} 张子图，尺寸 {canvas_w}×{canvas_h}）")

    return output_files


def merge_all_symbols(
    screenshot_results: dict,
    cfg: dict,
    date_str: str
) -> dict:
    """
    对所有标的进行合图
    返回：{symbol: [combined_path, ...]}
    """
    merge_cfg = cfg["merge"]
    intervals = cfg["intervals"]
    charts_root = Path(cfg["output"]["charts_dir"]) / date_str
    combined_map = {}

    for symbol, iv_paths in screenshot_results.items():
        sym_safe = sanitize_symbol(symbol)
        out_dir = str(charts_root / sym_safe)
        log(f"合并图片：{symbol}")
        combined = merge_images_for_symbol(
            symbol, iv_paths, intervals, merge_cfg, out_dir
        )
        combined_map[symbol] = combined

    return combined_map


# ─── 豆包/Sider 自动上传（半自动版）────────────────────────────────────────────

async def upload_to_doubao(page, images: list, prompt: str, symbol: str):
    """
    向豆包网页端上传图片并提交分析（半自动，供参考）
    豆包 URL: https://www.doubao.com/chat/
    """
    log(f"  → 上传至豆包：{symbol}（{len(images)} 张合图）")

    await page.goto("https://www.doubao.com/chat/")
    await page.wait_for_load_state("networkidle")
    await asyncio.sleep(2)

    # 找到上传按钮（selector 可能需要根据豆包实际DOM调整）
    upload_selectors = [
        "input[type='file']",
        "[aria-label='上传文件']",
        "[data-testid='upload-button']",
    ]

    for img_path in images:
        uploaded = False
        for sel in upload_selectors:
            try:
                inp = page.locator(sel).first
                if await inp.is_visible(timeout=1500):
                    await inp.set_input_files(img_path)
                    await asyncio.sleep(1)
                    uploaded = True
                    break
            except Exception:
                pass
        if not uploaded:
            log(f"  [WARN] 豆包上传失败：{img_path}，请手动上传")

    # 填入提示词
    textarea_sel = "textarea, [contenteditable='true'], [role='textbox']"
    try:
        ta = page.locator(textarea_sel).last
        await ta.click()
        await ta.fill(prompt)
        await asyncio.sleep(0.5)
        # 注意：不自动点发送，留给用户确认
        log(f"  ✓ 提示词已填入，请手动点击发送按钮")
    except Exception as e:
        log(f"  [ERROR] 填入提示词失败：{e}")


async def upload_to_sider(page, images: list, prompt: str, symbol: str):
    """
    向 Sider 网页端上传图片
    Sider URL: https://sider.ai/
    """
    log(f"  → 上传至 Sider：{symbol}（{len(images)} 张合图）")

    await page.goto("https://sider.ai/app")
    await page.wait_for_load_state("networkidle")
    await asyncio.sleep(2)

    upload_selectors = [
        "input[type='file']",
        "[aria-label='Upload']",
    ]

    for img_path in images:
        for sel in upload_selectors:
            try:
                inp = page.locator(sel).first
                if await inp.is_visible(timeout=1500):
                    await inp.set_input_files(img_path)
                    await asyncio.sleep(1)
                    break
            except Exception:
                pass

    try:
        ta = page.locator("textarea, [contenteditable='true']").last
        await ta.click()
        await ta.fill(prompt)
        log(f"  ✓ Sider 提示词已填入，请手动确认发送")
    except Exception as e:
        log(f"  [ERROR] Sider 填入失败：{e}")


# ─── 主流程 ───────────────────────────────────────────────────────────────────

async def run_screenshot_only(cfg: dict, date_str: str) -> dict:
    """只执行截图 + 合图"""
    log("=" * 60)
    log("阶段1：TradingView 截图")
    log("=" * 60)
    screenshot_results = await take_screenshots(cfg, date_str)

    log("=" * 60)
    log("阶段2：多周期合图")
    log("=" * 60)
    combined_map = merge_all_symbols(screenshot_results, cfg, date_str)

    return combined_map


async def run_upload(cfg: dict, combined_map: dict, target: str = "doubao"):
    """执行上传到 AI 网页端（半自动）"""
    from playwright.async_api import async_playwright

    tv_cfg = cfg["tradingview"]
    prompt = cfg["prompt"]["wave_analysis"]

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=tv_cfg["user_data_dir"] + "_ai",
            headless=False,
            viewport={"width": 1600, "height": 900},
        )
        page = context.pages[0] if context.pages else await context.new_page()

        for symbol, combined_paths in combined_map.items():
            if not combined_paths:
                continue
            if target == "doubao":
                await upload_to_doubao(page, combined_paths, prompt, symbol)
            elif target == "sider":
                await upload_to_sider(page, combined_paths, prompt, symbol)

            # 等待用户手动发送，给足时间
            log(f"  ⏳ 请在浏览器中手动点击发送，然后等待20秒继续下一个...")
            await asyncio.sleep(20)

        await context.close()


# ─── 批量从已有截图重新合图（不重新截图）────────────────────────────────────────

def remerge_from_existing(cfg: dict, date_str: str) -> dict:
    """
    扫描已有截图目录，重新生成合图（无需重新截图）
    适合：调整合图参数后重跑
    """
    charts_root = Path(cfg["output"]["charts_dir"]) / date_str
    intervals = cfg["intervals"]
    combined_map = {}

    for symbol in cfg["symbols"]:
        sym_safe = sanitize_symbol(symbol)
        sym_dir = charts_root / sym_safe
        if not sym_dir.exists():
            continue

        iv_paths = {}
        for iv in intervals:
            label = INTERVAL_LABELS.get(iv, iv)
            p = sym_dir / f"{sym_safe}_{label}.png"
            if p.exists():
                iv_paths[iv] = str(p)

        if iv_paths:
            log(f"重合图：{symbol}（找到 {len(iv_paths)} 个周期）")
            merged = merge_images_for_symbol(
                symbol, iv_paths, intervals,
                cfg["merge"], str(sym_dir)
            )
            combined_map[symbol] = merged

    return combined_map


# ─── 打印摘要 ─────────────────────────────────────────────────────────────────

def print_summary(combined_map: dict):
    log("=" * 60)
    log("📊 合图摘要")
    log("=" * 60)
    total_files = 0
    for symbol, paths in combined_map.items():
        status = f"✓ {len(paths)} 张合图" if paths else "✗ 无合图"
        print(f"  {symbol:<25} {status}")
        if paths:
            for p in paths:
                print(f"      → {p}")
        total_files += len(paths)
    log(f"\n  共 {len(combined_map)} 个标的，{total_files} 张合图")
    log("=" * 60)


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TradingView 自动截图 + 多周期合图 + AI 分析上传系统"
    )
    parser.add_argument(
        "--mode",
        choices=["screenshot", "merge", "upload", "all"],
        default="screenshot",
        help=(
            "screenshot : 截图+合图（完整流程）\n"
            "merge      : 仅重新合图（使用已有截图）\n"
            "upload     : 仅上传合图到 AI（需先截图）\n"
            "all        : 截图+合图+上传"
        )
    )
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--date", default=today_str(), help="日期（默认今天，格式 YYYY-MM-DD）")
    parser.add_argument("--target", choices=["doubao", "sider"], default="doubao", help="上传目标平台")
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="覆盖配置中的 symbols，例：--symbols NASDAQ:MU NASDAQ:NVDA"
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.symbols:
        cfg["symbols"] = args.symbols

    ensure_dirs(cfg["output"]["charts_dir"], cfg["output"]["results_dir"])

    if args.mode in ("screenshot", "all"):
        combined_map = asyncio.run(run_screenshot_only(cfg, args.date))
        print_summary(combined_map)

        # 保存合图路径索引
        index_path = Path(cfg["output"]["charts_dir"]) / args.date / "combined_index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(combined_map, f, ensure_ascii=False, indent=2)
        log(f"合图索引已保存：{index_path}")

        if args.mode == "all":
            asyncio.run(run_upload(cfg, combined_map, args.target))

    elif args.mode == "merge":
        combined_map = remerge_from_existing(cfg, args.date)
        print_summary(combined_map)

    elif args.mode == "upload":
        # 从已有索引加载合图路径
        index_path = Path(cfg["output"]["charts_dir"]) / args.date / "combined_index.json"
        if not index_path.exists():
            log("[ERROR] 找不到合图索引，请先运行 screenshot 模式")
            return
        with open(index_path, "r", encoding="utf-8") as f:
            combined_map = json.load(f)
        asyncio.run(run_upload(cfg, combined_map, args.target))


if __name__ == "__main__":
    main()
    # 首次：截图模式，手动登录后 Ctrl+C 中断即可
    # python tv_analyzer20260425_增加自动检测_避免白图_但没有zoom.py - -mode    screenshot - -symbols    NASDAQ: MU

    #仅截图 + 合图（推荐日常使用）
    # python  tv_analyzer20260425_增加自动检测_避免白图_但没有zoom.py - -mode    screenshot

    #只针对部分标的截图
    # python   tv_analyzer20260425_增加自动检测_避免白图_但没有zoom.py - -mode    screenshot - -symbols    NASDAQ: NVDA    NASDAQ: AMD    OANDA: XAUUSD

    #重新合图（不重新截图，用于调整合图参数后）
    # python tv_analyzer20260425_增加自动检测_避免白图_但没有zoom.py --mode merge

    #截图 + 合图 + 自动上传豆包（半自动，需手动点发送）
    # python tv_analyzer20260425_增加自动检测_避免白图_但没有zoom.py --mode all --target doubao

    #只上传（截图已存在时)
    # python tv_analyzer20260425_增加自动检测_避免白图_但没有zoom.py --mode upload --target sider

    #指定历史日期的已有截图重新合图
    # python tv_analyzer20260425_增加自动检测_避免白图_但没有zoom.py --mode merge --date 2026-04-24