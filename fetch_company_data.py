import argparse
import os
from typing import Dict, List

import pandas as pd

# 原始字段 -> 训练脚本字段名
FIELD_ALIAS: Dict[str, str] = {
    "stock_change_rate": "pct_chg",
    "stock_adjclose": "adj_close",
    "stock_turnover_rate": "turnover",
    "stock_amount": "amount",
}

DEFAULT_FIELDS: List[str] = list(FIELD_ALIAS.keys())


def _to_long(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """把 wide-format(行=日期, 列=股票) 统一转换成长表。"""
    tmp = pd.DataFrame(df).copy()

    # 如果日期不在列中，使用 index 作为日期
    if "date" in tmp.columns:
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.set_index("date")
    else:
        tmp.index = pd.to_datetime(tmp.index)

    long_df = tmp.stack(dropna=False).reset_index()
    long_df.columns = ["date", "ticker", field]
    long_df["ticker"] = long_df["ticker"].astype(str)
    return long_df


def _fetch_bulk(begin: str, end: str, fields: List[str]):
    """优先按你给的方式一次性抓取：fetch_general(begin=..., end=..., fields=[...])。"""
    from data_interface import fetch_general

    return fetch_general(begin=begin, end=end, fields=fields)


def _fetch_single_field(field: str, begin: str, end: str):
    """兼容不同 data_interface 版本的单字段签名。"""
    from data_interface import fetch_general

    kw_candidates = [
        {"field": field, "start_date": begin, "end_date": end},
        {"data_dict": field, "start_date": begin, "end_date": end},
        {"name": field, "start_date": begin, "end_date": end},
        {"factor": field, "start_date": begin, "end_date": end},
        {"field": field, "begin": begin, "end": end},
    ]
    for kwargs in kw_candidates:
        try:
            data = fetch_general(**kwargs)
            if data is not None:
                return data
        except TypeError:
            continue

    pos_candidates = [
        (field, begin, end),
        (field, begin, end, "D"),
    ]
    for args in pos_candidates:
        try:
            data = fetch_general(*args)
            if data is not None:
                return data
        except TypeError:
            continue

    raise RuntimeError(f"无法抓取字段 {field}，请确认 fetch_general 的参数签名。")


def build_dataset(begin: str, end: str, fields: List[str]) -> pd.DataFrame:
    """抓取并合并为标准长表：date,ticker,pct_chg,adj_close,turnover,amount。"""
    # 先尝试 bulk 模式（你原脚本使用方式）
    raw_dict = None
    try:
        raw_dict = _fetch_bulk(begin=begin, end=end, fields=fields)
        if not isinstance(raw_dict, dict):
            raw_dict = None
    except Exception:
        raw_dict = None

    if raw_dict is None:
        raw_dict = {}
        for f in fields:
            raw_dict[f] = _fetch_single_field(f, begin=begin, end=end)

    processed = []
    for f in fields:
        if f not in raw_dict:
            raise KeyError(f"返回结果缺少字段 {f}")
        processed.append(_to_long(raw_dict[f], f).set_index(["date", "ticker"]))

    final_df = pd.concat(processed, axis=1).reset_index()

    # 别名列（供 train_gnn.py 直接使用）
    for src, dst in FIELD_ALIAS.items():
        final_df[dst] = final_df[src]

    final_df = final_df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return final_df


def resolve_output_path(output: str, desktop_gnn: bool) -> str:
    if desktop_gnn:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        folder = os.path.join(desktop, "GNN")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, output)
    return output


def main():
    parser = argparse.ArgumentParser(description="从 data_interface.fetch_general 抓取股票长表")
    parser.add_argument("--start", default="2023-01-03", help="开始日期")
    parser.add_argument("--end", default="2024-01-03", help="结束日期")
    parser.add_argument("--output", default="stock_data_long_format.csv", help="输出文件名或路径")
    parser.add_argument("--desktop-gnn", action="store_true", help="保存到 ~/Desktop/GNN/ 下")
    args = parser.parse_args()

    try:
        from data_interface import fetch_general  # noqa: F401
        print("✅ 成功导入 data_interface 库")
    except ImportError as e:
        raise SystemExit(f"❌ 未找到 data_interface 库: {e}")

    print("🚀 正在抓取数据...")
    df = build_dataset(begin=args.start, end=args.end, fields=DEFAULT_FIELDS)

    out_path = resolve_output_path(args.output, args.desktop_gnn)
    df.to_csv(out_path, index=False, encoding="utf_8_sig")

    print("-" * 30)
    print("🚀 导出成功！格式：长表 (Long Format)")
    print(f"保存路径: {out_path}")
    print(f"行数: {len(df)}, 列: {list(df.columns)}")
    print(f"预览前10行:\n{df.head(10)}")
    print("-" * 30)


if __name__ == "__main__":
    main()
