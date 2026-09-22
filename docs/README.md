# Portfolio Risk Engine — 本地风险度量与压力测试

一个本地可运行的 FastAPI服务，提供多资产组合的敞口聚合、尾部风险度量
（历史模拟法 + 参数法）、可命名/可组合的压力情景、相关性假设校验、
参数集版本化与并发批量评估。

## 快速开始

```bash
uv sync                       # 安装依赖（含 pytest）
uv run pytest tests/ -q       # 运行全部测试（60 例）
uv run uvicorn main:app --reload   # 启动服务，默认 http://127.0.0.1:8000
```

接口文档（OpenAPI）：启动后访问 `http://127.0.0.1:8000/docs`。

### 示例：风险评估

```bash
curl -s -X POST http://127.0.0.1:8000/api/risk/assess \
  -H 'Content-Type: application/json' -d '{
  "portfolio": {"name": "fund-x",
    "holdings": [{"instrument": "A", "quantity": 10},
                 {"instrument": "B", "quantity": -5}]},
  "market": {"dates": ["2026-01-01", "2026-01-02", "2026-01-03"],
             "prices": {"A": [100, 101, 103], "B": [50, 51, 50]}},
  "window": 2, "confidence": 0.99, "horizon_days": 1
}'
```

### 示例：压力情景

```bash
curl -s -X POST http://127.0.0.1:8000/api/risk/scenarios/evaluate \
  -H 'Content-Type: application/json' -d '{
  "portfolio": {"name": "fund-x",
    "holdings": [{"instrument": "A", "quantity": 10}]},
  "market": {"dates": ["2026-01-01", "2026-01-02"],
             "prices": {"A": [100, 102]}},
  "scenario": {"name": "eq_drop",
    "shocks": [{"instrument": "A", "relative_move": -0.10}]}
}'
```

## 风险指标口径（约定）

所有结论**确定性可复现**：相同输入必得相同输出，不依赖随机数或第三方
数值库的内部实现。

### 收益率与敞口

- 简单收益率：`r[t] = p[t]/p[t-1] - 1`，逐日、逐标的。
- 敞口权重：`w_i = qty_i * price_i(last)`（以最新价计的市值敞口）。
- 组合每期盈亏：`PnL[t] = Σ_i w_i * r_i[t]`（金额口径）。
- `gross_exposure = Σ|w_i|`，`net_exposure = Σ w_i`。

### 历史模拟法（Historical）

- 取最近 `window` 期组合盈亏序列，取**亏损** `L = -PnL`。
- VaR 采用**下秩（lower-rank）分位数**：升序排序后取
  `index = ceil(confidence * n) - 1`。这是确定性的，不做插值。
- ES（Expected Shortfall）= 不小于 VaR 的亏损样本的算术平均。
- 带符号口径：若窗口内最差情形仍为盈利，VaR/ES 可为负（表示该置信
  水平下仍盈利），不做截断，以保持口径诚实、可解释。

### 参数法（Parametric）

- 基于样本协方差（无偏，`n-1`）的正态假设 VaR/ES。
- 组合波动率：`σ_p = sqrt(wᵀ Σ w)`，其中 `Σ = diag(σ) · C · diag(σ)`，
  `C` 为相关性假设（未提供时默认单位阵，即假设因子不相关）。
- `VaR = z(c) · σ_p`，`ES = φ(z(c))/(1-c) · σ_p`，`z` 为标准正态
  单侧分位数（内置高精度反函数，不依赖 scipy）。

### 期限换算

- 两类指标均按 `sqrt(horizon_days)` 缩放（平方根时间法则）。

## 情景与相关性假设

### 压力情景

- 情景 = 一组命名因子扰动 `FactorShock(instrument, relative_move)`。
- 情景 PnL 为**一阶线性**：`PnL = Σ_i qty_i * price_i * move_i`。
- **组合语义**：`compose_scenarios` 将多个情景的扰动按顺序**拼接**，
  同一因子的多次扰动全部保留并在求和时累加。由此保证：
  - 组合情景 PnL **逐位等于**各子情景独立评估之和（线性可加）；
  - 情景结论与逐因子扰动之和**自洽**（结果中
    `consistent_with_single_factor` 标记，容差 1e-9 相对量级）。
- 对未持有或无行情数据的标的施加扰动会被拒绝
  （`UNKNOWN_INSTRUMENT`）。

### 相关性/协方差校验

相关性矩阵按固定顺序校验，**任何一步失败即按原因拒绝，绝不静默修正**：

1. `DIMENSION_MISMATCH` — 行/列数与标的数不符，或假设的标的集合与
   组合标的不一致；
2. `INVALID_CORRELATION_VALUE` — 非有限值、超出 [-1,1]、对角线非 1；
3. `ASYMMETRIC_MATRIX` — 非对称（容差 1e-9）；
4. `NOT_POSITIVE_DEFINITE` — Cholesky 分解失败（非正定）。

底层数值实现的异常（如分解中的算术错误）会被包装为
`COMPUTATION_ERROR`，**不会**把原始异常透出给调用方。

## 版本兼容策略

- 每个风险报告与情景结论都打上 `version` 戳（默认 `v1`）。
- `RulesVersion` 显式声明 `compatible_parents`：只有被声明兼容的旧
  版本结论才允许在新规则下重算。
- `reassess_under_version(stamped_version, target_version)`：
  - 目标版本兼容旧版本 → 重算并以新版本重新打戳（结论应一致）；
  - 不兼容 → 拒绝并返回 `INCOMPATIBLE_VERSION`，除非调用方显式
    传 `allow_breaking=True`（此时结论以新版本打戳，表示口径已变）。
- 未注册的版本、未注册的情景/相关性名称分别返回
  `UNKNOWN_VERSION` / `UNKNOWN_SCENARIO`。

## 并发批量评估

- `POST /api/risk/batch` 或 `RiskService.batch_assess`。
- 保证：输出顺序与请求顺序一致；并发结果与串行完全一致；单个请求
  失败只影响其自身槽位，其余结论不受影响。
- 资源/时间上限：
  - 批量大小超限 → 整体拒绝 `BATCH_SIZE_LIMIT`（HTTP 429）；
  - 单项/整体超出时间预算 → 该槽位标记 `EVALUATION_TIMEOUT`。

## 错误码一览

| code | 含义 | HTTP |
|---|---|---|
| `INSUFFICIENT_DATA` | 历史不足/窗口超过样本/非正价格/序列不齐 | 422 |
| `INVALID_WINDOW` | window<1、horizon<1、置信度越界 | 422 |
| `UNKNOWN_INSTRUMENT` | 标的无行情或未持有 | 422 |
| `UNKNOWN_SCENARIO` | 情景/相关性名称未注册 | 422 |
| `UNKNOWN_VERSION` | 版本未注册 | 422 |
| `DIMENSION_MISMATCH` | 矩阵维度/标的集合不匹配 | 422 |
| `INVALID_CORRELATION_VALUE` | 相关值越界/非有限/对角非1 | 422 |
| `ASYMMETRIC_MATRIX` | 矩阵非对称 | 422 |
| `NOT_POSITIVE_DEFINITE` | 矩阵非正定 | 422 |
| `INCOMPATIBLE_VERSION` | 跨版本重算未声明兼容 | 422 |
| `BATCH_SIZE_LIMIT` | 批量超上限 | 429 |
| `EVALUATION_TIMEOUT` | 超时间预算 | 429 |
| `COMPUTATION_ERROR` | 内部数值失败（已包装） | 500 |
| `INTERNAL_ERROR` | 未预期错误（兜底，不透出细节） | 500 |

## 本地验证方法

```bash
# 全量测试（含非法相关性、极端情景、并发批量）
uv run pytest tests/ -q

# 查看每个用例的输入摘要与判定依据（日志）
uv run pytest tests/ -q -s -o log_cli=true --log-cli-level=INFO

# 单模块
uv run pytest tests/test_correlation.py -q
uv run pytest tests/test_batch.py -q
```

测试日志约定：每个用例通过 `log.info("case=...; criterion: ...")`
打印**输入摘要**与**判定依据**，失败时可直接定位是输入构造问题还是
结论偏离预期。
