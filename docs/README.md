# Portfolio Risk Engine — 本地风险度量与压力测试

一个本地可运行的 FastAPI服务，提供多资产组合的敞口聚合、尾部风险度量
（历史模拟法 + 参数法）、可命名/可组合的压力情景、相关性假设校验、
参数集版本化与并发批量评估。

## 快速开始

```bash
uv sync                       # 安装依赖（含 pytest）
uv run pytest tests/ -q       # 运行全部测试（64 例）
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

### 可命名、可组合的相关性假设

- `POST /api/risk/correlations` 注册命名相关性矩阵（注册时立即严格校验，
  非法矩阵当场拒绝）；
- `POST /api/risk/correlations/compose` 将多个**不相交因子块**做块状对角
  组合（跨块相关系数置 0）；若因子块存在重叠，因无法无歧义合并，按
  `DIMENSION_MISMATCH` 明确拒绝，不做平均或静默修正；
- 评估时可用内联 `correlation` 或已注册的 `correlation_name` 引用；两者
  同时提供返回 `INVALID_PARAMETER`；引用不存在的名称返回
  `UNKNOWN_ASSUMPTION`。批量中单个错误引用只影响其自身槽位。

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
  - 单项/整体超出时间预算 → 该槽位标记 `EVALUATION_TIMEOUT`；
  - 时限到点时**已完成的条目原样保留**，只有真正未完成的条目才
    被判为超时。

## 批量扫描（多组合 × 多已登记情景）

- `POST /api/risk/scan` 或 `RiskService.scan`：一次提交多个组合，每个
  组合对一组**已注册情景名**逐一求值。
- 展开顺序为「组合优先」：job 0 的全部情景、job 1 的全部情景……
  返回顺序与提交顺序严格一致，与并行度无关，且与逐个单跑的结果
  完全一致。
- 单条失败（如情景名未注册 `UNKNOWN_SCENARIO`、标的未持有
  `UNKNOWN_INSTRUMENT`）只占用自己的槽位，其余条目照常出结果。
- 条目数上限按**展开后的 (组合 × 情景) 对数**计：超限整批拒绝
  `BATCH_SIZE_LIMIT`；总时限到点未完成的对标记 `EVALUATION_TIMEOUT`，
  已完成的对保留。
- 四类失败可区分：超时 `EVALUATION_TIMEOUT`、超容量
  `BATCH_SIZE_LIMIT`、输入非法（如 `UNKNOWN_SCENARIO` 等 422 族）、
  内部错误 `COMPUTATION_ERROR`。

## 跨版本批量重算对比

- `POST /api/risk/reassess` 或 `RiskService.batch_reassess`：把一批按
  旧版本打戳的结论，在目标版本下整批重算并逐条判定：
  - `REPRODUCED` — 重算结论与原结论**逐位一致**（`version` 戳本身
    不参与比较），旧结论在新规则下仍成立；
  - `CHANGED` — 允许重算但结论已不同，口径实际发生了变化；
  - `REJECTED` — 不允许重算（版本未声明兼容且未显式授权、版本未
    注册、输入非法等），条目携带具体错误码。
- 单条的拒绝/失败不影响其他条目，也不打乱返回顺序。
- **破坏性升级绝不静默生效**：未声明兼容的版本对默认判 `REJECTED`
  （`INCOMPATIBLE_VERSION`）；只有调用方在该条目上显式传
  `allow_breaking=true` 才会重算，且结论以**新版本重新打戳**。

## 结果快照与最差情形排名

- `POST /api/risk/scan/snapshot` 或 `RiskService.scan_snapshot`：跑一次
  扫描并把结果冻结成一份自包含快照（`format:
  risk-engine-snapshot/1`）。
- **确定性**：相同输入重复跑任意多次、任意并行度，快照序列化结果
  逐位一致。需要字节级稳定导出时用
  `risk_engine.snapshot.canonical_json(snapshot)`（键排序、浮点按
  最短可还原表示、拒绝 NaN/Infinity）。
- 每条目携带：输入概况（组合/情景/持仓数/行情天数）、规则版本戳、
  判定理由（`rationale`，说明结论如何得出或为何被拒）。
- 排名（`rankings`）：
  - `worst_by_portfolio` — 每个组合在其所有情景下的最差（最小）
    PnL，按该值升序排（亏得最多的在前）；
  - `worst_by_scenario` — 每个情景在所有组合下的最差 PnL，同理；
  - 口径在载荷中明示：`metric: pnl`、`order: ascending`；并列时按
    **名称字典序、再按首次提交下标**（`tie_break` 字段说明），因此
    同一批输入每次排名完全相同。失败条目不参与排名。

## 批量上限与时限配置

`BatchLimits`（库）与请求体字段（HTTP）一一对应：

| 配置 | 默认 | 含义 |
|---|---|---|
| `max_items` | 128 | 条目数上限（扫描按展开后的对数计），超限整批拒绝 |
| `timeout_seconds` | 5.0 | 整批总时间预算，到点未完成条目标 `EVALUATION_TIMEOUT` |
| `max_workers` | 8 | 并行度（仅库内可调；不影响结果与顺序） |

HTTP 请求体示例（`/scan`、`/reassess`、`/batch` 同构）：

```json
{"jobs": [...], "max_items": 256, "timeout_seconds": 10.0}
```

## 错误码一览

| code | 含义 | HTTP |
|---|---|---|
| `INSUFFICIENT_DATA` | 历史不足/窗口超过样本/非正价格/序列不齐 | 422 |
| `INVALID_WINDOW` | window<1、horizon<1 | 422 |
| `INVALID_PARAMETER` | 置信度越界、冲突入参 | 422 |
| `UNKNOWN_INSTRUMENT` | 标的无行情或未持有 | 422 |
| `UNKNOWN_SCENARIO` | 情景名称未注册 | 422 |
| `UNKNOWN_ASSUMPTION` | 相关性假设名称未注册 | 422 |
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
# 全量测试（含非法相关性、极端情景、并发批量、扫描、跨版本对比、快照）
# 默认即打印每个用例的输入摘要与判定依据，无需额外参数
uv run pytest tests/ -q

# 单模块
uv run pytest tests/test_correlation.py -q
uv run pytest tests/test_batch.py -q
uv run pytest tests/test_scan.py -q
uv run pytest tests/test_reassess.py -q
uv run pytest tests/test_snapshot.py -q
```

测试日志约定：每个用例通过 `log.info("case=...; criterion: ...")`
打印**输入摘要**与**判定依据**（pytest.ini 已默认开启
`log_cli=true`，INFO 级），失败时可直接定位是输入构造问题还是
结论偏离预期。

## 与既有能力的兼容范围

- 新增的 `/scan`、`/scan/snapshot`、`/reassess` 端点与
  `RiskService.scan / scan_snapshot / batch_reassess` 均为**纯增量**，
  不改变 `/assess`、`/batch`、情景与相关性端点的既有行为。
- 快照格式独立版本化（`risk-engine-snapshot/1`），与规则版本
  （`v1`/`v2`/...）正交：规则版本戳在每条目内，快照格式版本在
  文档头部。
- 跨版本重算沿用既有 `compatible_parents` 声明机制；未声明兼容的
  版本对默认拒绝，授权后重算并以新版本打戳，旧结论不被改写。
