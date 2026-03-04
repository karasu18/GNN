import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv


@dataclass
class Config:
    lookback: int = 20
    hidden_dim: int = 96
    gat_hidden: int = 64
    heads: int = 4
    dropout: float = 0.2
    lr: float = 6e-4
    epochs: int = 80
    topk: int = 20

    reg_weight: float = 0.5
    ic_weight: float = 0.35
    rank_weight: float = 0.15

    weight_decay: float = 2e-4
    clip_grad: float = 1.0
    patience: int = 12
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        h = self.norm(h[-1])
        return self.drop(h)


class GNNPredictor(nn.Module):
    def __init__(self, input_dim: int, cfg: Config):
        super().__init__()
        self.temporal = TemporalEncoder(input_dim, cfg.hidden_dim, cfg.dropout)
        self.gat1 = GATConv(cfg.hidden_dim, cfg.gat_hidden, heads=cfg.heads, dropout=cfg.dropout)
        self.gat2 = GATConv(cfg.gat_hidden * cfg.heads, cfg.gat_hidden, heads=1, dropout=cfg.dropout)
        self.norm = nn.LayerNorm(cfg.gat_hidden)
        self.reg_head = nn.Sequential(
            nn.Linear(cfg.gat_hidden, cfg.gat_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.gat_hidden, 1),
        )

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.temporal(x_seq)
        h = F.elu(self.gat1(h, edge_index))
        h = F.elu(self.gat2(h, edge_index))
        h = self.norm(h)
        return self.reg_head(h).squeeze(-1)


def winsorize_by_train(y_list: List[np.ndarray], n_train: int, q: float = 0.01) -> Tuple[List[np.ndarray], float, float]:
    train_y = np.concatenate(y_list[:n_train], axis=0)
    lo, hi = np.nanquantile(train_y, q), np.nanquantile(train_y, 1 - q)
    out = [np.clip(y, lo, hi) for y in y_list]
    return out, float(lo), float(hi)


def build_correlation_graph(ret_matrix: np.ndarray, topk: int) -> torch.Tensor:
    corr = np.corrcoef(ret_matrix.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, -np.inf)

    edges = []
    n = corr.shape[0]
    topk = max(1, min(topk, n - 1))
    for i in range(n):
        nbr_idx = np.argpartition(corr[i], -topk)[-topk:]
        for j in nbr_idx:
            edges.append((i, j))
            edges.append((j, i))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = out.groupby("ticker")["adj_close"].pct_change(fill_method=None)
    out["log_turnover"] = np.log(out["turnover"].clip(lower=1e-6))
    out["log_amount"] = np.log1p(out["amount"].clip(lower=0))

    g = out.groupby("ticker")
    out["ma5_ret"] = g["ret"].rolling(5).mean().reset_index(level=0, drop=True)
    out["ma10_ret"] = g["ret"].rolling(10).mean().reset_index(level=0, drop=True)
    out["vol5_ret"] = g["ret"].rolling(5).std().reset_index(level=0, drop=True)
    out["vol10_ret"] = g["ret"].rolling(10).std().reset_index(level=0, drop=True)
    out["mom5"] = g["adj_close"].pct_change(5)
    out["mom10"] = g["adj_close"].pct_change(10)
    out["target"] = g["ret"].shift(-1)
    return out


def build_panel(df: pd.DataFrame, feat_cols: List[str], lookback: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    dates = sorted(df["date"].unique())
    tickers = sorted(df["ticker"].unique())

    pivot_feats = {}
    for c in feat_cols + ["target"]:
        pivot_feats[c] = df.pivot(index="date", columns="ticker", values=c).reindex(index=dates, columns=tickers)

    X_list, y_list, m_list = [], [], []
    for t in range(lookback, len(dates) - 1):
        x_window = []
        valid = np.ones(len(tickers), dtype=bool)
        for c in feat_cols:
            arr = pivot_feats[c].iloc[t - lookback:t].values
            valid &= np.isfinite(arr).all(axis=0)
            x_window.append(arr.T)

        x = np.stack(x_window, axis=-1)
        y = pivot_feats["target"].iloc[t].values
        valid &= np.isfinite(y)

        X_list.append(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        y_list.append(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0))
        m_list.append(valid)

    return X_list, y_list, m_list


def ic_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    vx = pred - pred.mean()
    vy = target - target.mean()
    denom = (vx.square().sum().sqrt() * vy.square().sum().sqrt()).clamp_min(eps)
    return (vx * vy).sum() / denom


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, sample_pairs: int = 2048) -> torch.Tensor:
    n = pred.shape[0]
    if n < 3:
        return pred.new_tensor(0.0)

    idx_i = torch.randint(0, n, (sample_pairs,), device=pred.device)
    idx_j = torch.randint(0, n, (sample_pairs,), device=pred.device)
    neq = idx_i != idx_j
    idx_i, idx_j = idx_i[neq], idx_j[neq]
    if idx_i.numel() == 0:
        return pred.new_tensor(0.0)

    y_diff = target[idx_i] - target[idx_j]
    p_diff = pred[idx_i] - pred[idx_j]
    sign = torch.sign(y_diff)
    valid = sign != 0
    if valid.sum() == 0:
        return pred.new_tensor(0.0)

    margin = 0.0
    return F.relu(margin - sign[valid] * p_diff[valid]).mean()


def train_one_epoch(model, optimizer, x_seq, y, mask, edge_index, cfg: Config):
    model.train()
    optimizer.zero_grad()
    ret_pred = model(x_seq, edge_index)
    if mask.sum() == 0:
        return 0.0

    yv = y[mask]
    rp = ret_pred[mask]

    loss_reg = F.huber_loss(rp, yv)
    loss_ic = 1.0 - ic_torch(rp, yv)
    loss_rank = pairwise_rank_loss(rp, yv)

    loss = cfg.reg_weight * loss_reg + cfg.ic_weight * loss_ic + cfg.rank_weight * loss_rank
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
    optimizer.step()
    return float(loss.item())


def evaluate_one(model, x_seq, y, mask, edge_index, threshold: float):
    model.eval()
    with torch.no_grad():
        ret_pred = model(x_seq, edge_index)
        if mask.sum() == 0:
            return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

        yv = y[mask]
        rp = ret_pred[mask]

        mse = F.mse_loss(rp, yv).item()
        pred_dir = rp > threshold
        true_dir = yv > 0
        acc = (pred_dir == true_dir).float().mean().item()

        pred_pos = pred_dir.float().mean().item()
        true_pos = true_dir.float().mean().item()

        rp_np = rp.cpu().numpy()
        yv_np = yv.cpu().numpy()
        if np.std(rp_np) < 1e-12 or np.std(yv_np) < 1e-12:
            ic = float("nan")
        else:
            ic = float(np.corrcoef(rp_np, yv_np)[0, 1])

    return mse, acc, pred_pos, true_pos, ic


def evaluate_range(model, X_list, y_list, m_list, edge_index, to_tensor, start, end, threshold: float):
    mses, accs, pred_pos, true_pos, ics = [], [], [], [], []
    for i in range(start, end):
        x, y, m = to_tensor(X_list[i], y_list[i], m_list[i])
        mse, acc, ppr, tpr, ic = evaluate_one(model, x, y, m, edge_index, threshold)
        if not np.isnan(mse):
            mses.append(mse)
            accs.append(acc)
            pred_pos.append(ppr)
            true_pos.append(tpr)
            if not np.isnan(ic):
                ics.append(ic)

    if len(mses) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    mean_ic = float(np.mean(ics)) if len(ics) else float("nan")
    return float(np.mean(mses)), float(np.mean(accs)), float(np.mean(pred_pos)), float(np.mean(true_pos)), mean_ic


def pick_best_threshold(model, X_list, y_list, m_list, edge_index, to_tensor, start, end) -> float:
    best_thr, best_score = 0.0, -1e9
    for thr in np.linspace(-0.01, 0.01, 41):
        _, acc, pred_pos, true_pos, ic = evaluate_range(
            model, X_list, y_list, m_list, edge_index, to_tensor, start, end, float(thr)
        )
        if np.isnan(acc):
            continue
        score = (0.5 * acc) + (0.5 * (0.0 if np.isnan(ic) else ic)) - 0.2 * abs(pred_pos - true_pos)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr


def main(args):
    cfg = Config(
        lookback=args.lookback,
        epochs=args.epochs,
        topk=args.topk,
        reg_weight=args.reg_weight,
        ic_weight=args.ic_weight,
        rank_weight=args.rank_weight,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
    )

    set_seed(cfg.seed)

    if not os.path.exists(args.csv):
        raise FileNotFoundError(
            f"未找到 CSV 文件: {args.csv}\n"
            "请先运行: python fetch_company_data.py --start 2023-01-03 --end 2024-01-03 --output stock_data_long_format.csv\n"
            "或在训练时显式指定: python train_gnn.py --csv <你的csv路径>"
        )

    df = pd.read_csv(args.csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    df = make_features(df)
    feat_cols = [
        "ret", "log_turnover", "log_amount",
        "ma5_ret", "ma10_ret", "vol5_ret", "vol10_ret", "mom5", "mom10",
    ]

    X_list, y_list, m_list = build_panel(df, feat_cols, cfg.lookback)
    n = len(X_list)
    n_train, n_valid = int(n * 0.7), int(n * 0.85)

    y_list, ylo, yhi = winsorize_by_train(y_list, n_train=n_train, q=args.winsor_q)
    print(f"winsorize target by train quantile q={args.winsor_q}: [{ylo:.6f}, {yhi:.6f}]")

    train_ret = []
    for i in range(n_train):
        ret_today = X_list[i][:, -1, 0].copy()
        ret_today[~m_list[i]] = np.nan
        train_ret.append(ret_today)
    train_ret = np.stack(train_ret, axis=0)

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
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=4)

    best_state = None
    best_ic = -1e9
    bad_epochs = 0

    for ep in range(cfg.epochs):
        train_losses = []
        for i in range(n_train):
            x, y, m = to_tensor(X_list[i], y_list[i], m_list[i])
            loss = train_one_epoch(model, optim, x, y, m, edge_index, cfg)
            train_losses.append(loss)

        vmse, vacc, vpred_pos, vtrue_pos, vic = evaluate_range(
            model, X_list, y_list, m_list, edge_index, to_tensor, n_train, n_valid, threshold=0.0
        )
        vic_for_scheduler = -1e6 if np.isnan(vic) else vic
        scheduler.step(vic_for_scheduler)

        if vic_for_scheduler > best_ic:
            best_ic = vic_for_scheduler
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"epoch={ep+1:02d} lr={optim.param_groups[0]['lr']:.2e} train_loss={np.mean(train_losses):.6f} "
            f"valid_mse={vmse:.6f} valid_dir_acc={vacc:.4f} "
            f"valid_pred_pos={vpred_pos:.4f} valid_true_pos={vtrue_pos:.4f} valid_ic={vic:.4f}"
        )

        if bad_epochs >= cfg.patience:
            print(f"early stop at epoch={ep+1}, best_valid_ic={best_ic:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    best_thr = pick_best_threshold(model, X_list, y_list, m_list, edge_index, to_tensor, n_train, n_valid)
    print(f"best_valid_threshold={best_thr:.4f}")

    vmse, vacc, vpred_pos, vtrue_pos, vic = evaluate_range(
        model, X_list, y_list, m_list, edge_index, to_tensor, n_train, n_valid, threshold=best_thr
    )
    print(
        f"final_valid_mse={vmse:.6f} final_valid_dir_acc={vacc:.4f} "
        f"final_valid_pred_pos={vpred_pos:.4f} final_valid_true_pos={vtrue_pos:.4f} final_valid_ic={vic:.4f}"
    )

    tmse, tacc, tpred_pos, ttrue_pos, tic = evaluate_range(
        model, X_list, y_list, m_list, edge_index, to_tensor, n_valid, n, threshold=best_thr
    )
    print(
        f"test_mse={tmse:.6f} test_dir_acc={tacc:.4f} "
        f"test_pred_pos={tpred_pos:.4f} test_true_pos={ttrue_pos:.4f} test_ic={tic:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="stock_data_long_format.csv", help="input data csv path")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--reg-weight", type=float, default=0.5)
    parser.add_argument("--ic-weight", type=float, default=0.35)
    parser.add_argument("--rank-weight", type=float, default=0.15)
    parser.add_argument("--winsor-q", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
