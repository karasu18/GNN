"""
train_gnn_commented.py
======================
这是 `train_gnn.py` 的“中文注释讲解版”。
目标：帮助你快速看懂每一步在做什么。

你可以直接运行：
    python train_gnn_commented.py --csv stock_data_long_format.csv
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv


# ========== 1) 配置区 ==========
# 统一放超参数，后续调参更方便。
@dataclass
class Config:
    lookback: int = 20      # 使用过去多少天作为输入窗口
    hidden_dim: int = 64    # GRU 隐层维度
    gat_hidden: int = 64    # GAT 每层输出通道
    heads: int = 4          # GAT 多头数量
    dropout: float = 0.2    # Dropout 比例
    lr: float = 1e-3        # 学习率
    epochs: int = 30        # 训练轮数
    topk: int = 15          # 每个节点保留相关性 Top-K 邻居


# ========== 2) 模型结构：时序编码 + 图编码 ==========
class TemporalEncoder(nn.Module):
    """把每只股票过去 L 天的序列编码成一个向量。"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, L, F]，N=股票数，L=lookback，F=特征数
        _, h = self.gru(x)
        return h[-1]  # [N, H]


class GNNPredictor(nn.Module):
    """先 GRU，再 GAT，再 MLP 输出 next-day return。"""

    def __init__(self, input_dim: int, cfg: Config):
        super().__init__()
        self.temporal = TemporalEncoder(input_dim, cfg.hidden_dim)
        self.gat1 = GATConv(cfg.hidden_dim, cfg.gat_hidden, heads=cfg.heads, dropout=cfg.dropout)
        self.gat2 = GATConv(cfg.gat_hidden * cfg.heads, cfg.gat_hidden, heads=1, dropout=cfg.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.gat_hidden, cfg.gat_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.gat_hidden, 1),
        )

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.temporal(x_seq)
        h = F.elu(self.gat1(h, edge_index))
        h = F.elu(self.gat2(h, edge_index))
        return self.mlp(h).squeeze(-1)  # [N]


# ========== 3) 建图 ==========
def build_correlation_graph(ret_matrix: np.ndarray, topk: int) -> torch.Tensor:
    """
    用训练期收益率相关性建图。
    ret_matrix: [T, N]，T=训练期天数，N=股票数
    """
    corr = np.corrcoef(ret_matrix.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, -np.inf)  # 不连自己

    edges = []
    n = corr.shape[0]
    topk = max(1, min(topk, n - 1))
    for i in range(n):
        nbr_idx = np.argpartition(corr[i], -topk)[-topk:]
        for j in nbr_idx:
            edges.append((i, j))
            edges.append((j, i))  # 双向边

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# ========== 4) 特征工程 ==========
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """从原始列生成训练用特征与目标。"""
    out = df.copy()

    # ret: 当日收益率。fill_method=None 用于避免 pandas FutureWarning。
    out["ret"] = out.groupby("ticker")["adj_close"].pct_change(fill_method=None)

    # 对数变换：减弱长尾分布
    out["log_turnover"] = np.log(out["turnover"].clip(lower=1e-6))
    out["log_amount"] = np.log1p(out["amount"].clip(lower=0))

    # 简单滚动统计特征
    out["ma5_ret"] = out.groupby("ticker")["ret"].rolling(5).mean().reset_index(level=0, drop=True)
    out["vol5_ret"] = out.groupby("ticker")["ret"].rolling(5).std().reset_index(level=0, drop=True)

    # 目标：下一天收益率
    out["target"] = out.groupby("ticker")["ret"].shift(-1)
    return out


# ========== 5) 面板构造 ==========
def build_panel(
    df: pd.DataFrame, feat_cols: List[str], lookback: int
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[pd.Timestamp]]:
    """
    返回：
      X_list: 每个时间点的输入 [N, L, F]
      y_list: 每个时间点目标 [N]
      m_list: 有效样本 mask [N]
      d_list: 对应日期
    """
    dates = sorted(df["date"].unique())
    tickers = sorted(df["ticker"].unique())

    pivot_feats = {}
    for c in feat_cols + ["target"]:
        pivot_feats[c] = df.pivot(index="date", columns="ticker", values=c).reindex(index=dates, columns=tickers)

    X_list, y_list, m_list, d_list = [], [], [], []
    for t in range(lookback, len(dates) - 1):
        x_window = []
        valid = np.ones(len(tickers), dtype=bool)

        # 从 t-lookback 到 t-1 组窗口
        for c in feat_cols:
            arr = pivot_feats[c].iloc[t - lookback:t].values  # [L, N]
            valid &= np.isfinite(arr).all(axis=0)
            x_window.append(arr.T)  # [N, L]

        x = np.stack(x_window, axis=-1)  # [N, L, F]
        y = pivot_feats["target"].iloc[t].values  # [N]
        valid &= np.isfinite(y)

        X_list.append(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        y_list.append(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0))
        m_list.append(valid)
        d_list.append(dates[t])

    return X_list, y_list, m_list, d_list


# ========== 6) 训练与评估 ==========
def train_one_epoch(model, optimizer, x_seq, y, mask, edge_index):
    model.train()
    optimizer.zero_grad()
    pred = model(x_seq, edge_index)

    # 只在有效标签上算 loss
    if mask.sum() == 0:
        return 0.0
    loss = F.huber_loss(pred[mask], y[mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


def evaluate(model, x_seq, y, mask, edge_index):
    model.eval()
    with torch.no_grad():
        pred = model(x_seq, edge_index)
        if mask.sum() == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")

        mse = F.mse_loss(pred[mask], y[mask]).item()
        direction = ((pred[mask] > 0) == (y[mask] > 0)).float().mean().item()

        # 诊断“总猜涨/总猜跌”偏置
        pred_pos_ratio = (pred[mask] > 0).float().mean().item()
        true_pos_ratio = (y[mask] > 0).float().mean().item()
    return mse, direction, pred_pos_ratio, true_pos_ratio


def evaluate_range(model, X_list, y_list, m_list, edge_index, to_tensor, start, end):
    """在一个时间区间上做平均评估，而不是只看某一天。"""
    mses, accs, pred_pos, true_pos = [], [], [], []
    for i in range(start, end):
        x, y, m = to_tensor(X_list[i], y_list[i], m_list[i])
        mse, acc, ppr, tpr = evaluate(model, x, y, m, edge_index)
        if not np.isnan(mse):
            mses.append(mse)
            accs.append(acc)
            pred_pos.append(ppr)
            true_pos.append(tpr)

    if len(mses) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return float(np.mean(mses)), float(np.mean(accs)), float(np.mean(pred_pos)), float(np.mean(true_pos))


# ========== 7) 主流程 ==========
def main(args):
    cfg = Config(lookback=args.lookback, epochs=args.epochs, topk=args.topk)

    if not os.path.exists(args.csv):
        raise FileNotFoundError(
            f"未找到 CSV 文件: {args.csv}\n"
            "请先运行: python fetch_company_data.py --start 2023-01-03 --end 2024-01-03 --output stock_data_long_format.csv\n"
            "或在训练时显式指定: python train_gnn_commented.py --csv <你的csv路径>"
        )

    # 读取数据（必须含有 date,ticker,adj_close,turnover,amount）
    df = pd.read_csv(args.csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    df = make_features(df)
    feat_cols = ["ret", "log_turnover", "log_amount", "ma5_ret", "vol5_ret"]

    X_list, y_list, m_list, _ = build_panel(df, feat_cols, cfg.lookback)
    n = len(X_list)
    n_train, n_valid = int(n * 0.7), int(n * 0.85)

    # 用训练区间构图
    train_ret = []
    for i in range(n_train):
        ret_today = X_list[i][:, -1, 0].copy()  # 窗口最后一天的 ret
        ret_today[~m_list[i]] = np.nan
        train_ret.append(ret_today)
    train_ret = np.stack(train_ret, axis=0)  # [T, N]

    # 用训练集拟合标准化器，避免泄漏
    scaler = StandardScaler()
    flat = np.concatenate([x.reshape(-1, x.shape[-1]) for x in X_list[:n_train]], axis=0)
    scaler.fit(flat)

    def to_tensor(x, y, m):
        x2 = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        return (
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(m, dtype=torch.bool),
        )

    edge_index = build_correlation_graph(train_ret, cfg.topk)
    model = GNNPredictor(input_dim=len(feat_cols), cfg=cfg)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # 训练循环
    for ep in range(cfg.epochs):
        train_losses = []
        for i in range(n_train):
            x, y, m = to_tensor(X_list[i], y_list[i], m_list[i])
            train_losses.append(train_one_epoch(model, optim, x, y, m, edge_index))

        vmse, vacc, vpred_pos, vtrue_pos = evaluate_range(
            model, X_list, y_list, m_list, edge_index, to_tensor, n_train, n_valid
        )
        print(
            f"epoch={ep+1:02d} "
            f"train_loss={np.mean(train_losses):.6f} "
            f"valid_mse={vmse:.6f} "
            f"valid_dir_acc={vacc:.4f} "
            f"valid_pred_pos={vpred_pos:.4f} "
            f"valid_true_pos={vtrue_pos:.4f}"
        )

    # 最终测试：整个 test 区间平均
    tmse, tacc, tpred_pos, ttrue_pos = evaluate_range(
        model, X_list, y_list, m_list, edge_index, to_tensor, n_valid, n
    )
    print(
        f"test_mse={tmse:.6f} test_dir_acc={tacc:.4f} "
        f"test_pred_pos={tpred_pos:.4f} test_true_pos={ttrue_pos:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="stock_data_long_format.csv", help="输入 CSV 路径")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--topk", type=int, default=15)
    main(parser.parse_args())
