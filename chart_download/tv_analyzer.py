"""
TradingView 自动截图 + 多周期合图 + 豆包/Sider 投喂系统
==========================================================
依赖安装：
    pip install playwright pillow pyyaml
    playwright install chromium

默认目录结构（相对于本脚本所在目录）：
    data_download/
        chart_download/
            tv_analyzer.py
            config.yaml
            charts/
                2026-04-25/
                    NASDAQ_MU/
                        NASDAQ_MU_1Y.png
                        NASDAQ_MU_1M.png
                        ...
                        NASDAQ_MU_combined.png
                    combined_index.json
                    combined_images_2026-04-25/
                    latest_combined_images_2026-04-25/
                    screening_report_2026-04-25.pdf
                    screening_sheets_2026-04-25/
                        screening_sheet_01.png
                        screening_sheet_02.png
            results/

说明：
1. 默认固定读取脚本同目录下的 config.yaml
2. 单标的分析继续使用每个标的自己的 *_combined.png
3. 批量筛选新增 PDF 报告输出
4. 不支持 PDF 的平台可使用 screening_sheet_01.png 这种总览拼图
5. PDF 首页会自动写入：prompt.screening / prompt.wave_analysis / symbols / intervals
"""

import asyncio
import io
import json
import shutil
import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageStat


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG = {
    "tradingview": {
        "base_url": "https://www.tradingview.com/chart/",
        "chart_id": "",
        "theme": "dark",
        "width": 1920,
        "height": 1080,
        "chart_area_top_offset": 60,
        "wait_after_load": 3000,
        "wait_after_interval": 2000,
        "user_data_dir": "./tv_profile",
        "navigation_timeout": 45000,
        "goto_retries": 3,
        "homepage_check": False,
        "proxy": "http://127.0.0.1:10809",
        "headless": False,
    },
    "intervals": ["12M", "3M", "1M", "1W", "1D", "60", "30", "15", "5"],
    "symbols": [
        "NASDAQ:NVDA", "NASDAQ:AMD", "NASDAQ:MU", "NASDAQ:MRVL", "NASDAQ:PLTR",
        "NASDAQ:META", "NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:AMZN", "NASDAQ:TSLA",
        "CME_MINI:NQ1!", "CME_MINI:ES1!", "NYSE:LLY", "NASDAQ:NDX",
        "OANDA:XAUUSD", "OANDA:XAGUSD", "NYMEX:CL1!", "SSE:000001", "SZSE:399006",
    ],
    "merge": {
        "enabled": True,
        "cols": 3,
        "thumb_width": 960,
        "label_height": 40,
        "title_height": 44,
        "font_size": 24,
        "background": "#1a1a2e",
        "max_per_sheet": 9,
        "output_suffix": "_combined",
    },
    "prompt": {
        "wave_analysis": """请基于我上传的 TradingView 蜡烛图进行波浪理论分析。

图片说明：
- 每张合图包含同一标的的多个周期（从年线到5分钟），从左到右、从上到下依次排列
- 请先从年线/月线建立大级别方向，再向下细化至日线、小时线和分钟线

分析要求：
1. 波浪分析至少细分四个层级浪型（周期级 / 主浪级 / 中级浪 / 小级浪）
2. 判断当前所处浪型结构（推动浪 / 调整浪，以及具体浪号）
3. 给出 2-3 个高概率结构路径，并说明触发条件和失效条件
4. 结合江恩和道氏原理，标出关键支撑/压力位
5. 判断该标的是否属于：刚启动(3浪初期) / 主升延长 / 5浪延伸 / 调整中 / 见顶嫌疑
6. 给出时间和目标价格的预期
7. 最终给出一个综合评分（1-10分）和买卖建议，和关注的位置点，什么时候、什么价格你估计的几种假设切换，或者概率倾斜。

请严格依据图片的价格数据分析，不要用网上的价格数据，每个路径请单独标注概率（高/中/低）。""",
        "screening": """以下是一组股票的多周期技术图表。请帮我做筛选评估：

对每个标的，请：
1. 标注当前所在浪型结构
2. 评估结构确定性（高/中/低）
3. 给出综合评分（1-10分）
4. 最终分类：A（强烈做多候选）/ B（考虑做多）/ C（观望）/ D（回避）

最后请汇总输出一个表格，列出：标的 | 浪型 | 评分 | 分类""",
    },
    "output": {"charts_dir": "./charts", "results_dir": "./results"},
}

INTERVAL_LABELS = {
    "12M": "1Y", "3M": "3M", "1M": "1M", "1W": "1W", "1D": "1D",
    "60": "1H", "30": "30", "15": "15", "5": "5", "1": "1",
}

CAPTURE_RETRIES = 3
BLANK_RECHECK_WAIT_SEC = 2.2
RETRY_BACKOFF_SEC = 1.2
SETTLE_CHECK_ROUNDS = 4
SETTLE_SLEEP_SEC = 0.6

PDF_PAGE_SIZE = (1654, 2339)
PDF_MARGIN = 70
PDF_TITLE_SIZE = 42
PDF_SECTION_SIZE = 24
PDF_BODY_SIZE = 18
PDF_LINE_GAP = 8

SHEET_CELL_WIDTH = 1500
SHEET_PADDING = 20
SHEET_TITLE_HEIGHT = 64
SHEET_LABEL_HEIGHT = 52


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def sanitize_symbol(symbol: str) -> str:
    return symbol.replace(":", "_").replace("!", "_main")


def deep_merge_dict(defaults: dict, actual: dict) -> dict:
    merged = deepcopy(defaults)
    for k, v in (actual or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge_dict(merged[k], v)
        else:
            merged[k] = v
    return merged


def resolve_path(path_value, base_dir: Path) -> str:
    p = Path(path_value)
    return str(p if p.is_absolute() else (base_dir / p).resolve())


def normalize_config_paths(cfg: dict, config_path: Path) -> dict:
    base_dir = config_path.parent.resolve()
    cfg["tradingview"]["user_data_dir"] = resolve_path(cfg["tradingview"]["user_data_dir"], base_dir)
    cfg["output"]["charts_dir"] = resolve_path(cfg["output"]["charts_dir"], base_dir)
    cfg["output"]["results_dir"] = resolve_path(cfg["output"]["results_dir"], base_dir)
    cfg["_meta"] = {"config_path": str(config_path.resolve()), "config_dir": str(base_dir)}
    return cfg


def load_config(path=None) -> dict:
    config_path = SCRIPT_DIR / "config.yaml" if path is None else Path(path)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    ensure_dirs(str(config_path.parent))
    if not config_path.exists():
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(DEFAULT_CONFIG, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"[CONFIG] 已生成默认配置文件：{config_path}，请按需修改后重新运行。")
    with open(config_path, "r", encoding="utf-8") as f:
        file_cfg = yaml.safe_load(f) or {}
    return normalize_config_paths(deep_merge_dict(DEFAULT_CONFIG, file_cfg), config_path)


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


def get_date_root(cfg: dict, date_str: str) -> Path:
    return Path(cfg["output"]["charts_dir"]) / date_str


def get_font(size: int, bold: bool = False):
    paths = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for fp in paths:
        try:
            if Path(fp).exists():
                return ImageFont.truetype(fp, size)
        except Exception:
            pass
    return ImageFont.load_default()


def font_line_height(font, fallback: int) -> int:
    return getattr(font, "size", fallback) + PDF_LINE_GAP


def text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    if not text:
        return 0
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int):
    text = "" if text is None else str(text).replace("\r\n", "\n").replace("\r", "\n")
    out = []
    for raw in text.split("\n"):
        if raw == "":
            out.append("")
            continue
        cur = ""
        for ch in raw:
            if not cur or text_width(draw, cur + ch, font) <= max_width:
                cur += ch
            else:
                out.append(cur)
                cur = ch
        if cur:
            out.append(cur)
    return out or [""]


def safe_open_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def extract_combined_order(path: Path) -> int:
    import re
    m = re.search(r"_combined(?:_(\d+))?\.png$", path.name, re.IGNORECASE)
    return int(m.group(1)) if m and m.group(1) else (0 if m else -1)


def find_latest_combined_image(symbol_dir: Path):
    files = [p for p in symbol_dir.glob("*_combined*.png") if p.is_file()]
    if not files:
        return None
    files.sort(key=lambda p: (extract_combined_order(p), p.name.lower()))
    return files[-1]


def prepare_flat_png_dir(target_dir: Path):
    ensure_dirs(str(target_dir))
    for old_png in target_dir.glob("*.png"):
        try:
            old_png.unlink()
        except Exception:
            pass


def copy_png_to_flat_dir(src_png: Path, target_dir: Path) -> Path:
    dst = target_dir / src_png.name
    if not dst.exists():
        shutil.copy2(src_png, dst)
        return dst
    idx = 2
    while True:
        candidate = target_dir / f"{src_png.stem}_{idx}{src_png.suffix}"
        if not candidate.exists():
            shutil.copy2(src_png, candidate)
            return candidate
        idx += 1


def scan_symbol_dirs(charts_root: Path):
    excluded = ("combined_images_", "latest_combined_images_", "screening_sheets_")
    return [p for p in sorted(charts_root.iterdir()) if p.is_dir() and not p.name.startswith(excluded)] if charts_root.exists() else []


def get_latest_combined_pairs(charts_root: Path, cfg: dict):
    pairs, seen = [], set()
    for symbol in cfg.get("symbols", []):
        sym_dir = charts_root / sanitize_symbol(symbol)
        if sym_dir.exists():
            latest = find_latest_combined_image(sym_dir)
            if latest:
                pairs.append((symbol, latest))
                seen.add(sym_dir.name)
    for sym_dir in scan_symbol_dirs(charts_root):
        if sym_dir.name in seen:
            continue
        latest = find_latest_combined_image(sym_dir)
        if latest:
            pairs.append((sym_dir.name, latest))
    return pairs


def format_symbols_text(symbols: list) -> str:
    return "\n".join([f"{i + 1}. {s}" for i, s in enumerate(symbols)])


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


def update_url_interval(url: str, interval: str) -> str:
    import urllib.parse
    tv_map = {"12M": "12M", "3M": "3M", "1M": "M", "1W": "W", "1D": "D", "60": "60", "30": "30", "15": "15", "5": "5", "1": "1"}
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    params["interval"] = [tv_map.get(interval, interval)]
    return parsed._replace(query=urllib.parse.urlencode({k: v[0] for k, v in params.items()})).geturl()


async def dismiss_popups(page):
    for sel in ["button[aria-label='Close']", ".tv-dialog__close", ".modal-dialog__close", "[data-role='toast-close']", ".js-dialog-close", "[aria-label='Cancel']"]:
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=600):
                await btn.click()
                await asyncio.sleep(0.4)
        except Exception:
            pass


async def focus_chart_area(page, tv_cfg: dict):
    try:
        await page.mouse.click(tv_cfg["width"] // 2, max(tv_cfg.get("chart_area_top_offset", 60) + 120, 280))
        await asyncio.sleep(0.15)
    except Exception:
        pass


async def wait_for_chart_settle(page, tv_cfg: dict, extra_wait_sec: float = 0.0):
    await asyncio.sleep(tv_cfg["wait_after_interval"] / 1000 + extra_wait_sec)
    for _ in range(SETTLE_CHECK_ROUNDS):
        await dismiss_popups(page)
        await focus_chart_area(page, tv_cfg)
        await asyncio.sleep(SETTLE_SLEEP_SEC)


async def capture_cropped_image(page, tv_cfg: dict) -> Image.Image:
    img = Image.open(io.BytesIO(await page.screenshot(full_page=False))).convert("RGB")
    top = tv_cfg.get("chart_area_top_offset", 60)
    return img.crop((0, top, img.width, img.height))


def is_probably_blank_chart(img: Image.Image) -> bool:
    try:
        w, h = img.size
        if w < 300 or h < 200:
            return True
        region = img.crop((int(w * 0.18), int(h * 0.12), int(w * 0.82), int(h * 0.82))).convert("L").resize((180, 100))
        stat = ImageStat.Stat(region)
        hist = region.histogram()
        total = region.width * region.height
        near_white = sum(hist[245:256]) / total
        near_light = sum(hist[230:256]) / total
        return (near_white > 0.94 and stat.stddev[0] < 10) or (near_light > 0.97 and stat.stddev[0] < 12) or (stat.mean[0] > 235 and stat.stddev[0] < 9)
    except Exception:
        return False


async def save_validated_chart(page, out_path: str, tv_cfg: dict, debug_tag: str = "") -> bool:
    img = await capture_cropped_image(page, tv_cfg)
    if not is_probably_blank_chart(img):
        img.save(out_path, "PNG", optimize=True)
        return True
    log(f"    [WARN] {debug_tag} 首次截图疑似空白，追加等待后复检")
    await asyncio.sleep(BLANK_RECHECK_WAIT_SEC)
    await dismiss_popups(page)
    await focus_chart_area(page, tv_cfg)
    img = await capture_cropped_image(page, tv_cfg)
    if not is_probably_blank_chart(img):
        img.save(out_path, "PNG", optimize=True)
        return True
    return False


async def set_interval_and_screenshot(page, symbol: str, symbol_url: str, interval: str, label: str, out_path: str, tv_cfg: dict) -> bool:
    new_url = update_url_interval(symbol_url, interval)
    for attempt in range(1, CAPTURE_RETRIES + 1):
        try:
            if attempt > 1:
                log(f"    [RETRY] {symbol} @ {label} 第 {attempt}/{CAPTURE_RETRIES} 次重试")
            await safe_goto(page, new_url, timeout_ms=tv_cfg.get("navigation_timeout", 45000), retries=1 if attempt > 1 else tv_cfg.get("goto_retries", 3))
            await wait_for_chart_settle(page, tv_cfg, extra_wait_sec=(attempt - 1) * RETRY_BACKOFF_SEC)
            if await save_validated_chart(page, out_path, tv_cfg, f"{symbol} @ {label}"):
                return True
            log(f"    [WARN] {symbol} @ {label} 仍疑似空白图，准备重试")
        except Exception as e:
            log(f"    [ERROR] 截图失败 {symbol} @ {label} (第{attempt}次): {e}")
        if attempt < CAPTURE_RETRIES:
            await asyncio.sleep(RETRY_BACKOFF_SEC * attempt)
            await dismiss_popups(page)
    return False


async def take_screenshots(cfg: dict, date_str: str):
    from playwright.async_api import async_playwright
    tv_cfg = cfg["tradingview"]
    charts_root = get_date_root(cfg, date_str)
    ensure_dirs(str(charts_root), cfg["output"]["results_dir"], tv_cfg["user_data_dir"])
    results = {}
    async with async_playwright() as p:
        launch_kwargs = {
            "user_data_dir": tv_cfg["user_data_dir"],
            "headless": tv_cfg.get("headless", False),
            "viewport": {"width": tv_cfg["width"], "height": tv_cfg["height"]},
            "args": ["--disable-blink-features=AutomationControlled", "--start-maximized"],
        }
        proxy_settings = get_proxy_settings(tv_cfg)
        if proxy_settings:
            launch_kwargs["proxy"] = proxy_settings
            log(f"[PROXY] 已启用代理：{proxy_settings.get('server')}")
        else:
            log("[PROXY] 未启用代理")
        context = await p.chromium.launch_persistent_context(**launch_kwargs)
        page = context.pages[0] if context.pages else await context.new_page()
        page.set_default_navigation_timeout(tv_cfg.get("navigation_timeout", 45000))
        page.set_default_timeout(tv_cfg.get("navigation_timeout", 45000))
        if tv_cfg.get("homepage_check", False):
            await safe_goto(page, tv_cfg.get("base_url", "https://www.tradingview.com/chart/"), timeout_ms=tv_cfg.get("navigation_timeout", 45000), retries=tv_cfg.get("goto_retries", 3))
            await asyncio.sleep(2)
        for symbol in cfg["symbols"]:
            sym_safe = sanitize_symbol(symbol)
            sym_dir = charts_root / sym_safe
            ensure_dirs(str(sym_dir))
            results[symbol] = {}
            chart_id = tv_cfg.get("chart_id", "")
            symbol_url = f"https://www.tradingview.com/chart/{chart_id}/?symbol={symbol}" if chart_id else f"https://www.tradingview.com/chart/?symbol={symbol}"
            log(f"  → 打开标的：{symbol}")
            await safe_goto(page, symbol_url, timeout_ms=tv_cfg.get("navigation_timeout", 45000), retries=tv_cfg.get("goto_retries", 3))
            await asyncio.sleep(tv_cfg["wait_after_load"] / 1000)
            await dismiss_popups(page)
            await focus_chart_area(page, tv_cfg)
            for interval in cfg["intervals"]:
                label = INTERVAL_LABELS.get(interval, interval)
                out_path = str(sym_dir / f"{sym_safe}_{label}.png")
                log(f"    截图：{symbol} @ {label}")
                if await set_interval_and_screenshot(page, symbol, symbol_url, interval, label, out_path, tv_cfg):
                    results[symbol][interval] = out_path
                else:
                    log(f"    [WARN] 最终失败：{symbol} @ {interval}")
            log(f"  ✓ {symbol} 截图完成（{len(results[symbol])}/{len(cfg['intervals'])} 个周期）")
        await context.close()
    return results


def merge_images_for_symbol(symbol: str, interval_paths: dict, intervals: list, merge_cfg: dict, out_dir: str, date_str: str) -> list:
    ordered = [(INTERVAL_LABELS.get(iv, iv), interval_paths[iv]) for iv in intervals if iv in interval_paths]
    if not ordered:
        log(f"  [WARN] {symbol} 无可用截图，跳过合图")
        return []
    cols = merge_cfg["cols"]
    thumb_w = merge_cfg["thumb_width"]
    label_h = merge_cfg["label_height"]
    title_h = merge_cfg.get("title_height", 44)
    font = get_font(merge_cfg["font_size"], bold=True)
    batches = [ordered[i:i + merge_cfg["max_per_sheet"]] for i in range(0, len(ordered), merge_cfg["max_per_sheet"])]
    outputs = []
    for batch_idx, batch in enumerate(batches):
        rows = (len(batch) + cols - 1) // cols
        try:
            sample = safe_open_image(Path(batch[0][1]))
            thumb_h = int(thumb_w * (sample.height / sample.width))
        except Exception:
            thumb_h = int(thumb_w * 9 / 16)
        canvas = Image.new("RGB", (cols * thumb_w, title_h + rows * (thumb_h + label_h)), merge_cfg["background"])
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, canvas.width, title_h], fill="#060614")
        draw.text((10, 8), f"{symbol}  —  多周期波浪分析图  （{date_str}）  第{batch_idx + 1}张/共{len(batches)}张", fill="#88ccff", font=font)
        for idx, (label, img_path) in enumerate(batch):
            row, col = divmod(idx, cols)
            x, y = col * thumb_w, title_h + row * (thumb_h + label_h)
            try:
                canvas.paste(safe_open_image(Path(img_path)).resize((thumb_w, thumb_h), Image.LANCZOS), (x, y))
            except Exception as e:
                log(f"  [WARN] 无法读取图片 {img_path}: {e}")
                canvas.paste(Image.new("RGB", (thumb_w, thumb_h), "#333344"), (x, y))
            draw.rectangle([x, y + thumb_h, x + thumb_w, y + thumb_h + label_h], fill="#0d0d1a")
            draw.text((x + 10, y + thumb_h + 6), f"{sanitize_symbol(symbol)}  |  {label}", fill="#e0e0e0", font=font)
        suffix = f"{merge_cfg['output_suffix']}_{batch_idx + 1}" if len(batches) > 1 else merge_cfg["output_suffix"]
        out_path = str(Path(out_dir) / f"{sanitize_symbol(symbol)}{suffix}.png")
        canvas.save(out_path, "PNG", optimize=True)
        outputs.append(out_path)
        log(f"  ✓ 合图保存：{out_path}")
    return outputs


def merge_all_symbols(screenshot_results: dict, cfg: dict, date_str: str) -> dict:
    charts_root = get_date_root(cfg, date_str)
    combined_map = {}
    for symbol, iv_paths in screenshot_results.items():
        log(f"合并图片：{symbol}")
        combined_map[symbol] = merge_images_for_symbol(symbol, iv_paths, cfg["intervals"], cfg["merge"], str(charts_root / sanitize_symbol(symbol)), date_str)
    return combined_map


def remerge_from_existing(cfg: dict, date_str: str) -> dict:
    charts_root = get_date_root(cfg, date_str)
    combined_map = {}
    for symbol in cfg["symbols"]:
        sym_safe = sanitize_symbol(symbol)
        sym_dir = charts_root / sym_safe
        if not sym_dir.exists():
            continue
        iv_paths = {}
        for iv in cfg["intervals"]:
            p = sym_dir / f"{sym_safe}_{INTERVAL_LABELS.get(iv, iv)}.png"
            if p.exists():
                iv_paths[iv] = str(p)
        if iv_paths:
            log(f"重合图：{symbol}（找到 {len(iv_paths)} 个周期）")
            combined_map[symbol] = merge_images_for_symbol(symbol, iv_paths, cfg["intervals"], cfg["merge"], str(sym_dir), date_str)
    return combined_map


def collect_all_combined_images_to_flat_dir(charts_root: Path, date_str: str) -> str:
    target_dir = charts_root / f"combined_images_{date_str}"
    prepare_flat_png_dir(target_dir)
    count = 0
    for item in scan_symbol_dirs(charts_root):
        files = sorted([p for p in item.glob("*_combined*.png") if p.is_file()], key=lambda p: (extract_combined_order(p), p.name.lower()))
        for png in files:
            copy_png_to_flat_dir(png, target_dir)
            count += 1
    log(f"  ✓ 全部 combined 图片已汇总到文件夹：{target_dir}（共 {count} 张合图）")
    return str(target_dir)


def collect_latest_combined_images_to_flat_dir(charts_root: Path, date_str: str) -> str:
    target_dir = charts_root / f"latest_combined_images_{date_str}"
    prepare_flat_png_dir(target_dir)
    count = 0
    for item in scan_symbol_dirs(charts_root):
        latest = find_latest_combined_image(item)
        if latest:
            copy_png_to_flat_dir(latest, target_dir)
            count += 1
    log(f"  ✓ 每个标的最后一张 combined 图片已汇总到文件夹：{target_dir}（共 {count} 个标的）")
    return str(target_dir)


def build_intro_pdf_pages(cfg: dict, date_str: str):
    page_w, page_h = PDF_PAGE_SIZE
    margin = PDF_MARGIN
    title_font = get_font(PDF_TITLE_SIZE, True)
    section_font = get_font(PDF_SECTION_SIZE, True)
    body_font = get_font(PDF_BODY_SIZE, False)
    blocks = [
        ("meta", f"日期：{date_str}"),
        ("meta", f"配置文件：{cfg.get('_meta', {}).get('config_path', '')}"),
        ("meta", f"intervals：{', '.join([str(x) for x in cfg.get('intervals', [])])}"),
        #("section", "symbols"),
        #("body", format_symbols_text(cfg.get("symbols", []))),
        ("section", "prompt.screening"),
        ("body", cfg.get("prompt", {}).get("screening", "")),
        ("section", "prompt.wave_analysis"),
        ("body", cfg.get("prompt", {}).get("wave_analysis", "")),
    ]
    pages, page_index, page, draw, y = [], 0, None, None, 0

    def new_page():
        nonlocal page_index, page, draw, y
        page_index += 1
        page = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(page)
        y = margin
        title = "TradingView 批量筛选 PDF 报告" + (f"（续 {page_index}）" if page_index > 1 else "")
        draw.text((margin, y), title, fill="black", font=title_font)
        y += font_line_height(title_font, PDF_TITLE_SIZE) + 10
        draw.line((margin, y, page_w - margin, y), fill="#666666", width=2)
        y += 24

    new_page()
    for kind, content in blocks:
        font = section_font if kind == "section" else body_font
        line_h = font_line_height(font, PDF_BODY_SIZE if kind != "section" else PDF_SECTION_SIZE)
        for line in wrap_text(draw, content, font, page_w - margin * 2):
            if y + line_h > page_h - margin:
                pages.append(page)
                new_page()
            draw.text((margin, y), line, fill="black", font=font)
            y += line_h + (4 if line == "" else 0)
        y += 10 if kind == "section" else 6
    pages.append(page)
    return pages


def build_symbol_pdf_page(symbol: str, image_path: Path, date_str: str):
    page_w, page_h = PDF_PAGE_SIZE
    margin = PDF_MARGIN
    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)
    title_font = get_font(30, True)
    meta_font = get_font(18, False)
    draw.text((margin, margin), f"{symbol}  —  多周期合图", fill="black", font=title_font)
    draw.text((margin, margin + 42), f"日期：{date_str}", fill="#444444", font=meta_font)
    draw.line((margin, margin + 76, page_w - margin, margin + 76), fill="#666666", width=2)
    try:
        chart = safe_open_image(image_path)
        chart.thumbnail((page_w - margin * 2, page_h - 200), Image.LANCZOS)
        page.paste(chart, ((page_w - chart.width) // 2, 110 + max(0, (page_h - 200 - chart.height) // 2)))
        draw.text((margin, page_h - 38), image_path.name, fill="#666666", font=meta_font)
    except Exception as e:
        draw.text((margin, 140), f"图片读取失败：{image_path}", fill="red", font=meta_font)
        draw.text((margin, 172), str(e), fill="red", font=meta_font)
    return page


def create_pdf_report(cfg: dict, date_str: str):
    charts_root = get_date_root(cfg, date_str)
    if not charts_root.exists():
        log(f"[ERROR] 找不到日期目录：{charts_root}")
        return None
    pairs = get_latest_combined_pairs(charts_root, cfg)
    if not pairs:
        log("[ERROR] 没找到可用于生成 PDF 的 combined 图片")
        return None
    pages = build_intro_pdf_pages(cfg, date_str)
    for symbol, img_path in pairs:
        pages.append(build_symbol_pdf_page(symbol, img_path, date_str))
    out_path = charts_root / f"screening_report_{date_str}.pdf"
    pages[0].save(str(out_path), "PDF", save_all=True, append_images=pages[1:], resolution=150.0)
    log(f"  ✓ PDF 报告已生成：{out_path}")
    return str(out_path)


def create_screening_sheets(cfg: dict, date_str: str, per_image: int = 4, cols: int = 2):
    charts_root = get_date_root(cfg, date_str)
    if not charts_root.exists():
        log(f"[ERROR] 找不到日期目录：{charts_root}")
        return []
    pairs = get_latest_combined_pairs(charts_root, cfg)
    if not pairs:
        log("[ERROR] 没找到可用于生成 screening sheet 的 combined 图片")
        return []
    per_image = max(1, int(per_image))
    cols = max(1, min(int(cols), per_image))
    target_dir = charts_root / f"screening_sheets_{date_str}"
    prepare_flat_png_dir(target_dir)
    outputs = []
    batches = [pairs[i:i + per_image] for i in range(0, len(pairs), per_image)]
    for batch_idx, batch in enumerate(batches, start=1):
        rows = (len(batch) + cols - 1) // cols
        try:
            sample = safe_open_image(batch[0][1])
            cell_h = int(SHEET_CELL_WIDTH * sample.height / sample.width)
        except Exception:
            cell_h = int(SHEET_CELL_WIDTH * 9 / 16)
        canvas_w = cols * SHEET_CELL_WIDTH + (cols + 1) * SHEET_PADDING
        canvas_h = SHEET_TITLE_HEIGHT + rows * (cell_h + SHEET_LABEL_HEIGHT) + (rows + 1) * SHEET_PADDING
        canvas = Image.new("RGB", (canvas_w, canvas_h), "#111827")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, canvas_w, SHEET_TITLE_HEIGHT], fill="#030712")
        draw.text((18, 14), f"TradingView Screening Sheet  {batch_idx:02d}/{len(batches)}   日期：{date_str}", fill="#93c5fd", font=get_font(28, True))
        label_font = get_font(22, False)
        for idx, (symbol, img_path) in enumerate(batch):
            row, col = divmod(idx, cols)
            x = SHEET_PADDING + col * (SHEET_CELL_WIDTH + SHEET_PADDING)
            y = SHEET_TITLE_HEIGHT + SHEET_PADDING + row * (cell_h + SHEET_LABEL_HEIGHT + SHEET_PADDING)
            try:
                canvas.paste(safe_open_image(img_path).resize((SHEET_CELL_WIDTH, cell_h), Image.LANCZOS), (x, y))
            except Exception as e:
                canvas.paste(Image.new("RGB", (SHEET_CELL_WIDTH, cell_h), "#374151"), (x, y))
                draw.text((x + 20, y + 20), f"图片读取失败\n{img_path.name}\n{e}", fill="#fca5a5", font=label_font)
            draw.rectangle([x, y + cell_h, x + SHEET_CELL_WIDTH, y + cell_h + SHEET_LABEL_HEIGHT], fill="#0f172a")
            draw.text((x + 12, y + cell_h + 12), f"{symbol}   |   {img_path.name}", fill="#e5e7eb", font=label_font)
        out_path = target_dir / f"screening_sheet_{batch_idx:02d}.png"
        canvas.save(out_path, "PNG", optimize=True)
        outputs.append(str(out_path))
        log(f"  ✓ 总览拼图已生成：{out_path}")
    return outputs


def print_summary(combined_map: dict):
    log("=" * 60)
    log("📊 合图摘要")
    log("=" * 60)
    total = 0
    for symbol, paths in combined_map.items():
        print(f"  {symbol:<25} {'✓ ' + str(len(paths)) + ' 张合图' if paths else '✗ 无合图'}")
        for p in paths:
            print(f"      → {p}")
        total += len(paths)
    log(f"\n  共 {len(combined_map)} 个标的，{total} 张合图")
    log("=" * 60)


async def run_screenshot_only(cfg: dict, date_str: str) -> dict:
    log("=" * 60)
    log("阶段1：TradingView 截图")
    log("=" * 60)
    screenshot_results = await take_screenshots(cfg, date_str)
    log("=" * 60)
    log("阶段2：多周期合图")
    log("=" * 60)
    return merge_all_symbols(screenshot_results, cfg, date_str)


async def upload_to_doubao(page, images: list, prompt: str, symbol: str):
    log(f"  → 上传至豆包：{symbol}（{len(images)} 张合图）")
    await page.goto("https://www.doubao.com/chat/")
    await page.wait_for_load_state("networkidle")
    await asyncio.sleep(2)
    for img_path in images:
        uploaded = False
        for sel in ["input[type='file']", "[aria-label='上传文件']", "[data-testid='upload-button']"]:
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
    try:
        ta = page.locator("textarea, [contenteditable='true'], [role='textbox']").last
        await ta.click()
        await ta.fill(prompt)
        log("  ✓ 提示词已填入，请手动点击发送按钮")
    except Exception as e:
        log(f"  [ERROR] 填入提示词失败：{e}")


async def upload_to_sider(page, images: list, prompt: str, symbol: str):
    log(f"  → 上传至 Sider：{symbol}（{len(images)} 张合图）")
    await page.goto("https://sider.ai/app")
    await page.wait_for_load_state("networkidle")
    await asyncio.sleep(2)
    for img_path in images:
        for sel in ["input[type='file']", "[aria-label='Upload']"]:
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
        log("  ✓ Sider 提示词已填入，请手动确认发送")
    except Exception as e:
        log(f"  [ERROR] Sider 填入失败：{e}")


async def run_upload(cfg: dict, combined_map: dict, target: str = "doubao"):
    from playwright.async_api import async_playwright
    tv_cfg = cfg["tradingview"]
    launch_kwargs = {"user_data_dir": tv_cfg["user_data_dir"] + "_ai", "headless": False, "viewport": {"width": 1600, "height": 900}}
    proxy_settings = get_proxy_settings(tv_cfg)
    if proxy_settings:
        launch_kwargs["proxy"] = proxy_settings
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(**launch_kwargs)
        page = context.pages[0] if context.pages else await context.new_page()
        for symbol, combined_paths in combined_map.items():
            if not combined_paths:
                continue
            if target == "doubao":
                await upload_to_doubao(page, combined_paths, cfg["prompt"]["wave_analysis"], symbol)
            else:
                await upload_to_sider(page, combined_paths, cfg["prompt"]["wave_analysis"], symbol)
            log("  ⏳ 请在浏览器中手动点击发送，然后等待20秒继续下一个...")
            await asyncio.sleep(20)
        await context.close()


def save_combined_index(cfg: dict, date_str: str, combined_map: dict):
    index_path = get_date_root(cfg, date_str) / "combined_index.json"
    ensure_dirs(str(index_path.parent))
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(combined_map, f, ensure_ascii=False, indent=2)
    log(f"合图索引已保存：{index_path}")


def generate_reports(cfg: dict, date_str: str, sheet_per_image: int, sheet_cols: int):
    pdf_path = create_pdf_report(cfg, date_str)
    sheet_paths = create_screening_sheets(cfg, date_str, per_image=sheet_per_image, cols=sheet_cols)
    return pdf_path, sheet_paths


def main():
    parser = argparse.ArgumentParser(description="TradingView 自动截图 + 多周期合图 + AI 分析上传系统")
    parser.add_argument("--mode", choices=["screenshot", "merge", "upload", "all", "collect", "zip", "pdf", "sheet", "report"], default="screenshot")
    parser.add_argument("--config", default=None, help="配置文件路径；不传则默认使用脚本同目录下 config.yaml")
    parser.add_argument("--date", default=today_str(), help="日期（默认今天，格式 YYYY-MM-DD）")
    parser.add_argument("--target", choices=["doubao", "sider"], default="doubao", help="上传目标平台")
    parser.add_argument("--symbols", nargs="+", help="覆盖配置中的 symbols")
    parser.add_argument("--sheet-per-image", type=int, default=4, help="每张 screening_sheet 放多少个标的，默认 4")
    parser.add_argument("--sheet-cols", type=int, default=2, help="screening_sheet 每张几列，默认 2")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.symbols:
        cfg["symbols"] = args.symbols
    ensure_dirs(cfg["output"]["charts_dir"], cfg["output"]["results_dir"], cfg["tradingview"]["user_data_dir"])

    print("[DEBUG] config path =", cfg["_meta"]["config_path"])
    print("[DEBUG] charts dir   =", cfg["output"]["charts_dir"])
    print("[DEBUG] results dir  =", cfg["output"]["results_dir"])
    print("[DEBUG] user_data_dir=", cfg["tradingview"]["user_data_dir"])
    print("[DEBUG] proxy        =", cfg["tradingview"].get("proxy"))
    print("[DEBUG] intervals    =", cfg.get("intervals"))
    print("[DEBUG] max_per_sheet=", cfg.get("merge", {}).get("max_per_sheet"))

    if args.mode in ("screenshot", "all"):
        combined_map = asyncio.run(run_screenshot_only(cfg, args.date))
        print_summary(combined_map)
        save_combined_index(cfg, args.date, combined_map)
        charts_root = get_date_root(cfg, args.date)
        collect_all_combined_images_to_flat_dir(charts_root, args.date)
        collect_latest_combined_images_to_flat_dir(charts_root, args.date)
        generate_reports(cfg, args.date, args.sheet_per_image, args.sheet_cols)
        if args.mode == "all":
            asyncio.run(run_upload(cfg, combined_map, args.target))

    elif args.mode == "merge":
        combined_map = remerge_from_existing(cfg, args.date)
        print_summary(combined_map)
        save_combined_index(cfg, args.date, combined_map)
        charts_root = get_date_root(cfg, args.date)
        collect_all_combined_images_to_flat_dir(charts_root, args.date)
        collect_latest_combined_images_to_flat_dir(charts_root, args.date)
        generate_reports(cfg, args.date, args.sheet_per_image, args.sheet_cols)

    elif args.mode in ("collect", "zip"):
        charts_root = get_date_root(cfg, args.date)
        if not charts_root.exists():
            log(f"[ERROR] 找不到日期目录：{charts_root}")
            return
        collect_all_combined_images_to_flat_dir(charts_root, args.date)
        collect_latest_combined_images_to_flat_dir(charts_root, args.date)
        generate_reports(cfg, args.date, args.sheet_per_image, args.sheet_cols)

    elif args.mode == "pdf":
        create_pdf_report(cfg, args.date)

    elif args.mode == "sheet":
        create_screening_sheets(cfg, args.date, per_image=args.sheet_per_image, cols=args.sheet_cols)

    elif args.mode == "report":
        generate_reports(cfg, args.date, args.sheet_per_image, args.sheet_cols)

    elif args.mode == "upload":
        index_path = get_date_root(cfg, args.date) / "combined_index.json"
        if not index_path.exists():
            log("[ERROR] 找不到合图索引，请先运行 screenshot 或 merge 模式")
            return
        with open(index_path, "r", encoding="utf-8") as f:
            combined_map = json.load(f)
        asyncio.run(run_upload(cfg, combined_map, args.target))


if __name__ == "__main__":
    main()
    # 只针对部分标的截图
    # python chart_download/tv_analyzer.py --mode screenshot --symbols GOOG CME_MINI:YM1!
    #  python chart_download/tv_analyzer.py --mode screenshot --symbols  CME_MINI:NQ1! CME_MINI:ES1! HKEX:HTI1! HKEX:HSI1! HKEx:3690 SSE:000001 SZSE:399006 SZSE:399005 CBOT_MINI:YM1!

    # 仅截图 + 合图 + 汇总 + PDF + screening sheet
    # python chart_download/tv_analyzer.py --mode screenshot

    # 重新合图（不重新截图）并重建 PDF / screening sheet
    # python chart_download/tv_analyzer.py --mode merge --date 2026-04-27

    # 仅生成 PDF
    # python chart_download/tv_analyzer.py --mode pdf --date 2026-04-25

    # 仅生成总览拼图，每张放 4 个标的，2 列
    # python data_download/chart_download/tv_analyzer.py --mode sheet --date 2026-04-25 --sheet-per-image 4 --sheet-cols 2

    # 同时生成 PDF + 总览拼图
    # python data_download/chart_download/tv_analyzer.py --mode report --date 2026-04-25

    # 截图 + 合图 + 上传豆包（半自动）
    # python data_download/chart_download/tv_analyzer.py --mode all --target doubao
