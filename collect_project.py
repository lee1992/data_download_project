import os
import sys
from pathlib import Path
import csv
import json

# ---------- 可调整的配置 ----------
MAX_FILE_SIZE_KB = 500        # 超过此大小的文件只展示元数据，不显示全部内容
MAX_CSV_ROWS = 5              # CSV 最多显示的行数
MAX_LOG_LINES = 30            # 日志文件最多显示尾部行数
SKIP_DIRS = {                 # 要跳过的目录（不递归）
    '__pycache__', '.git', '.idea', '.vscode', 'venv', 'env',
    'node_modules', 'build', 'dist', '.mypy_cache', '.pytest_cache',
    'egg-info', 'logs', 'data'   # 如果 logs/data 很大，可以暂时跳过
}
SKIP_EXTENSIONS = {           # 要跳过的文件后缀
    '.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe',
    '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.gz', '.tar',
    '.pkl', '.pickle', '.db', '.sqlite', '.sqlite3'
}
# ---------------------------------

def get_dir_tree(root, prefix=''):
    """生成缩进目录树字符串，自动跳过无法访问的条目"""
    try:
        entries = sorted(root.iterdir())
    except PermissionError:
        return [f"{prefix}[权限不足，无法列举 {root.name}]"]
    except OSError as e:
        return [f"{prefix}[列举失败: {root.name} - {e}]"]

    # 过滤掉需要跳过的目录名
    entries = [e for e in entries if e.name not in SKIP_DIRS]
    lines = []
    for i, entry in enumerate(entries):
        connector = '└── ' if i == len(entries) - 1 else '├── '
        if entry.suffix.lower() in SKIP_EXTENSIONS:
            continue   # 跳过不需要的后缀

        # 判断是否为目录（可能因权限或损坏而抛出异常）
        try:
            is_dir = entry.is_dir()
        except OSError:
            lines.append(f"{prefix}{connector}{entry.name} [无法判断类型，已跳过]")
            continue

        if is_dir:
            lines.append(f"{prefix}{connector}{entry.name}/")
            sub_prefix = '    ' if i == len(entries) - 1 else '│   '
            lines.extend(get_dir_tree(entry, prefix + sub_prefix))
        else:
            lines.append(f"{prefix}{connector}{entry.name}")
    return lines

def read_file_safe(path, max_size_kb=MAX_FILE_SIZE_KB):
    """安全读取文本文件，过大的文件只返回大小信息"""
    size_kb = path.stat().st_size / 1024
    if size_kb > max_size_kb:
        return f"[文件过大，未读取内容，大小: {size_kb:.1f} KB]"
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"[读取失败: {e}]"

def summarize_csv(path):
    """显示CSV的表头和前几行数据"""
    size_kb = path.stat().st_size / 1024
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            rows = []
            for i, row in enumerate(reader):
                if i < MAX_CSV_ROWS:
                    rows.append(row)
                else:
                    break
        total_rows_approx = "行数未知（仅读取前几行）"
        # 简单尝试统计行数（非严谨，对大文件快速估算）
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                line_count = sum(1 for _ in f) - 1  # 减去表头
            total_rows_approx = f"约 {line_count} 行"
        except:
            pass
        summary = f"CSV 文件摘要: {path.name}\n大小: {size_kb:.1f} KB, 总{total_rows_approx}\n列名: {headers}\n前{len(rows)}行数据:"
        for r in rows:
            summary += f"\n  {r}"
        return summary
    except Exception as e:
        return f"[CSV 读取失败: {e}]"

def summarize_log(path):
    """显示日志文件的最后若干行"""
    size_kb = path.stat().st_size / 1024
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            # 只取尾部行，避免加载整个大日志
            lines = f.readlines()
            tail = lines[-MAX_LOG_LINES:] if len(lines) > MAX_LOG_LINES else lines
        return f"日志文件: {path.name}  大小: {size_kb:.1f} KB, 显示最后{min(MAX_LOG_LINES, len(lines))}行:\n{''.join(tail)}"
    except Exception as e:
        return f"[日志读取失败: {e}]"

def process_project(root_dir):
    """主逻辑：遍历项目，生成汇总文本"""
    root = Path(root_dir).resolve()
    summary = []
    summary.append(f"项目路径: {root}")
    summary.append("=" * 60)

    # 1. 整个项目的目录树
    summary.append("【项目目录结构】")
    tree = get_dir_tree(root)
    summary.append(root.name + "/")
    summary.extend(tree)
    summary.append("")

    # 2. 遍历所有文件，分类处理
    for dirpath, dirnames, filenames in os.walk(root):
        # 跳过忽略的目录
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            file_path = Path(dirpath) / fname
            ext = file_path.suffix.lower()
            if ext in SKIP_EXTENSIONS:
                continue
            rel_path = file_path.relative_to(root)

            # Python 文件：完整内容
            if ext == '.py':
                content = read_file_safe(file_path)
                summary.append(f"\n{'='*60}")
                summary.append(f"Python 文件: {rel_path}")
                summary.append(f"{'='*60}")
                summary.append(content)

            # 配置文件（json, yaml, toml, ini 等）
            elif ext in {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'}:
                content = read_file_safe(file_path, max_size_kb=200)
                summary.append(f"\n{'='*60}")
                summary.append(f"配置文件: {rel_path}")
                summary.append(f"{'='*60}")
                summary.append(content)

            # CSV 文件：摘要信息
            elif ext == '.csv':
                summary.append(f"\n{'='*60}")
                summary.append(f"数据文件: {rel_path}")
                summary.append(f"{'='*60}")
                summary.append(summarize_csv(file_path))

            # 日志文件：尾部样本
            elif ext == '.log' or 'log' in fname.lower():
                summary.append(f"\n{'='*60}")
                summary.append(f"日志文件: {rel_path}")
                summary.append(f"{'='*60}")
                summary.append(summarize_log(file_path))

    return '\n'.join(summary)

if __name__ == '__main__':
    # 默认扫描当前工作目录，也可以从命令行参数传入路径
    target = sys.argv[1] if len(sys.argv) > 1 else '.'
    output_path = 'project_summary.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(process_project(target))
    print(f"汇总完成！请将 {output_path} 的内容发送给助手。")