# 基于 GNN 的股票价格预测（3000 股票 × 244 天）


## 0. 先把公司库数据抓成 CSV

你提到数据在公司库 `data_interface`，接口是 `fetch_general()`，字段为：

- `stock_change_rate`
- `stock_adjclose`
- `stock_turnover_rate`
- `stock_amount`

仓库里新增了 `fetch_company_data.py`，默认就抓你指定的日期区间 `2023-01-03` 到 `2024-01-03`，并输出训练所需长表：

```bash
python fetch_company_data.py \
  --start 2023-01-03 \
  --end 2024-01-03 \
  --output stock_panel_2023_2024.csv
```

输出列会统一成：`date,ticker,pct_chg,adj_close,turnover,amount`。

> 说明：不同公司库版本里 `fetch_general()` 参数名可能略有不同，脚本已做多种签名自动兼容。

这个仓库给出一个可落地的 GNN 时序预测基线，针对你当前的数据规模：

- 股票数量：约 3000
- 时间长度：244 个交易日
- 每日特征：涨跌幅、复权收盘价、换手率、成交额

## 1. 问题定义

建议先做 **T+1 收益率回归**（比直接预测绝对价格更稳定）：

- 输入：第 `t` 天及其过去 `lookback` 天的特征
- 输出：第 `t+1` 天收益率（`adj_close[t+1] / adj_close[t] - 1`）

后续再从收益率还原价格：

`pred_price[t+1] = adj_close[t] * (1 + pred_return[t+1])`

## 2. 图结构怎么建（关键）

你有 3000 只股票，单日图是 3000 个节点。边可以用以下方式构建：

1. **行业边**：同一行业全连接或 KNN 连接（最稳）
2. **相关性边**：用训练窗口内收益率皮尔逊相关系数，保留 Top-K
3. **流动性边（可选）**：按成交额/换手率相似度建边

实践中建议：

- 先用「行业边 + 相关性 Top-K」
- 每个节点限制 10~30 个邻居，避免图过密

## 3. 特征工程建议

在原始 4 列基础上加一些简单 alpha：

- `log_turnover = log(turnover + 1e-6)`
- `log_amount = log(amount + 1)`
- `ma5_ret`, `ma10_ret`（收益均值）
- `vol5_ret`, `vol10_ret`（收益波动）
- `rank_amount`（截面成交额分位数）

并做标准化：

- 时间维：仅用训练集统计量（防止泄漏）
- 截面维：每天做横截面 z-score 也常见

## 4. 模型结构（推荐起步）

建议用 **时序编码 + 图编码**：

- 每只股票过去 `lookback` 天序列 -> GRU/TemporalConv 编码
- 当天所有股票嵌入 -> GAT/GCN 做截面信息传播
- MLP 输出 next-day return

一个轻量基线：

- GRU(hidden=64)
- 2 层 GATConv(head=4)
- Dropout=0.2

## 5. 数据切分

时间序列必须按时间切分：

- Train: 前 70%
- Valid: 中间 15%
- Test: 最后 15%

不能随机打乱日期。

## 6. 评估指标

建议同时看：

- 回归误差：MSE / MAE
- 方向正确率：`sign(pred) == sign(true)`
- 截面 IC：每日 Pearson/Spearman IC，再看 IC 均值
- 策略回测（可选）：多空分组收益

## 7. 训练注意事项

- 目标值做 winsorize（极值截断）更稳定
- 使用 HuberLoss 往往比 MSE 更抗异常点
- 学习率 1e-3 起步，早停 patience 10
- 建图时只用训练窗口信息（避免未来数据泄漏）

## 8. 快速开始

1. 先运行 `python fetch_company_data.py --start 2023-01-03 --end 2024-01-03 --output stock_panel_2023_2024.csv` 抓取 CSV
2. 安装依赖：`torch`, `torch_geometric`, `pandas`, `numpy`, `scikit-learn`
3. 运行 `python train_gnn.py --csv stock_panel_2023_2024.csv`（或把文件命名为 `stock_data_long_format.csv` 后直接 `python train_gnn.py`）
   - 可选：`--reg-weight 0.5 --ic-weight 0.35 --rank-weight 0.15 --winsor-q 0.01 --lr 6e-4 --patience 12 --seed 42`（IC+排序联合优化版本）
> 训练日志里会打印 `valid_pred_pos`/`valid_true_pos`，用于观察模型是否过度偏向预测上涨或下跌。

> `train_gnn.py` 是可运行模板：包含数据处理、图构建、模型、训练和评估骨架。你可以先跑通，再逐步替换为行业/因子等更强特征。


## 9. 中文注释讲解版脚本

- 新增 `train_gnn_commented.py`，与 `train_gnn.py` 逻辑一致，但加入了更详细的中文注释，便于学习和二次开发。
- 运行方式：`python train_gnn_commented.py --csv stock_panel_2023_2024.csv`


## 10. 更稳的优化版训练脚本

- 新增 `train_gnn_v2.py`：在 `train_gnn.py` 基础上加入 `EMA` 参数平滑与更强排序损失，目标是提升训练稳定性与IC鲁棒性。
- 运行方式：`python train_gnn_v2.py --csv stock_panel_2023_2024.csv`
