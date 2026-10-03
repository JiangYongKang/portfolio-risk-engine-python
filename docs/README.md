# Portfolio Risk Engine — 本地风险度量与压力测试

一个本地可运行的 FastAPI服务，提供多资产组合的敞口聚合、尾部风险度量
（历史模拟法 + 参数法）、可命名/可组合的压力情景、相关性假设校验、
参数集版本化与并发批量评估，并支持**组合×情景批量扫描**、
**跨版本结论重算对比**与**确定性结果快照/最差情形排名**。

## 快速开始

```bash
uv sync                       # 安装依赖（含 pytest）
uv run pytest                 # 运行全部测试（89 例，默认即打印判定依据）
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
  - 单项/整体超出时间预算 → 该槽位标记 `EVALUATION_TIMEOUT`。
- **时限到点时已完成的结果原样保留**：只会把当时仍未完成的槽位判为
  超时，已经算完的条目既不丢弃也不会被误判成超时（见
  `test_completed_items_preserved_when_deadline_expires`）。

## 批量扫描：多个组合 × 多个已登记情景

日常"一批组合对一批登记好的情景做扫描"的能力，复用单条情景评估的
**同一条代码路径**，因此扫描结果与逐条单独评估**完全一致**。

- 端点：
  - `POST /api/risk/scan`：提交 `portfolios`（一批组合，共享一份行情）
    与 `scenario_names`（一批**已登记**情景名），服务端按
    **组合在外、情景在内**的固定顺序展开成笛卡尔积并扫描；
  - `RiskService.cross_product_scan(...)` / `RiskService.scan(cells)`
    提供同等能力；`cross_product_cells(...)` 也可只展开、不计算。
  - 需要每个组合用不同行情时，用 `ScanCell` 显式指定单元格。
- 顺序保证：返回顺序严格等于提交顺序（`index` 即提交位次），并行不
  乱序；不同 worker 数结果一致。
- 隔离保证：某个情景名不存在（`UNKNOWN_SCENARIO`）、相关性名不存在
  （`UNKNOWN_ASSUMPTION`）、组合不持有被冲击标的
  （`UNKNOWN_INSTRUMENT`）等，只影响该单元格，其余照常出结果。
- 容量与时限（每条请求可覆盖）：
  - `max_items`（默认 **256** 个单元格）：超限**整批拒绝**
    `BATCH_SIZE_LIMIT`，并说明"扫描有 N 个单元格、上限为 M"；
  - `timeout_seconds`（默认 **10s**）：总处理时限；到时仅未完成的槽位
    标记 `EVALUATION_TIMEOUT`，已完成的数值保留；
  - `max_workers`（默认 8）：并发度，不影响数值与顺序。
- 每条成功结果额外带 `input`（输入概况：组合、持仓数、情景、相关性、
  样本天数、版本）与 `rationale`（判定理由：线性一阶 PnL、盈亏方向、
  单因子自洽标记）。

```bash
curl -s -X POST http://127.0.0.1:8000/api/risk/scan \
  -H 'Content-Type: application/json' -d '{
  "portfolios": [
    {"name":"f1","holdings":[{"instrument":"A","quantity":10}]},
    {"name":"f2","holdings":[{"instrument":"B","quantity":-5}]}],
  "market": {"dates":["2026-01-01","2026-01-02","2026-01-03"],
             "prices":{"A":[100,101,103],"B":[50,51,50]}},
  "scenario_names": ["eq_drop","credit"],
  "max_items": 256, "timeout_seconds": 10
}'
```

## 跨版本批量重算对比

规则升级后，把一批**旧版本结论**放到新版本下整批重算并逐条判定。

- 端点：`POST /api/risk/recompare`
  （`RiskService.recompare_conclusions(...)`）。每条入参携带组合、行情、
  `stamped_version`（旧版本戳）与历史结论 `conclusion`，全批统一指定
  `target_version` 与 `allow_breaking`。
- 三种判定（逐槽位、顺序固定、互不影响）：
  - `REPRODUCED`（结论没变/可复现）：允许重算且新结论与旧结论
    **逐位一致**（比较时忽略 `version` 戳，兼容升级下的重新打戳不算
    变化）；返回新旧两份结论；
  - `CHANGED`（口径已不一样）：只有显式 `allow_breaking=true` 点头后
    才会真正按新口径重算，且数值确实变化时标记为 CHANGED；旧结论
    **原样保留、绝不被静默改写**，新旧并列返回，新结论打目标版本戳；
  - `REJECTED`（不允许重算/被拒/失败）：未获批准的破坏性升级
    （`INCOMPATIBLE_VERSION`）、未知版本、未知标的、内部计算失败等都
    归此判定；槽位带原始错误码，`new_conclusion` 为空，旧结论保留。
- 一条不能重算或中途失败，不影响其他条目的判定，也不打乱顺序。
- 容量/时限同批量：`max_items` 默认 256、`timeout_seconds` 默认 10s，
  超限整批拒绝 `BATCH_SIZE_LIMIT`。

## 结果快照与最差情形排名

把一次扫描固化为**本地双击可开、内容稳定**的结果快照。

- 端点：`POST /api/risk/scan/snapshot`
  （`RiskService.export_scan_snapshot(...)` / `build_snapshot(...)`），
  返回**自包含单文件 HTML**（无外部资源，内嵌一份规范 JSON 数据岛），
  可直接保存为 `.html` 本地打开。
- **逐位确定性**：同一批输入重复导出任意次，快照字节完全一致——不含
  时间戳/主机名等环境信息；dict 按键排序、固定分隔符输出（见
  `canonical_json`）；底层数值本身就是确定性的（同输入同输出）。
- 每行包含：提交位次 `index`、成败、组合与情景名、**版本标记**、
  PnL、**输入概况 `input`**、**判定理由 `rationale`**（失败行为错误码
  与原因）。
- 两个方向的**最差情形排名**：
  - `rank_by_portfolio`：每个组合在所有情景下的最差 PnL；
  - `rank_by_scenario`：每个情景在所有组合下的最差 PnL。
  - 口径在快照中明确标注：指标 `ranking_metric = min_scenario_pnl`
    （"最差"= PnL 最小/亏损最大，盈利记为负亏损）；
  - 并列规则 `tie_break`：先按最差 PnL 升序，再按名称字典序，再按
    对手方（情景/组合）名称字典序——保证同批输入每次排名完全一样。
  - 仅成功单元格参与排名；失败单元格仍在明细表中列出。

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
- 批量跨版本重算对比（见上"跨版本批量重算对比"一节）在此基础上把每条
  归类为 `REPRODUCED` / `CHANGED` / `REJECTED`，不会静默改写历史结论。

## 与现有能力/版本的兼容范围

- 新增模块（`scan.py`、`recompare.py`、`snapshot.py`）均为**增量**：
  原有的 `/assess`、`/batch`、`/scenarios/*`、`/correlations/*` 端点，
  数据契约（`RiskReport`、`ScenarioResult`、`BatchItemResult`）与错误码
  保持不变，旧用例无回归。
- 扫描复用 `run_batch` 与 `evaluate_scenario`，故扫描单元格结果与
  单条 `/scenarios/evaluate` 结果同源、数值一致。
- 重算对比只读取既有 `VersionRegistry` 的兼容声明与 `assess` 口径，
  默认规则仍为 `v1`；未注册 `v2/v3` 时不会产生新的版本行为。
- 快照为只读导出物，不改变任何计算结果，也不依赖外部网络/资源。

## 失败如何区分

扫描/重算/批量中，四类情况彼此可区分：

| 情况 | 表现 | 错误码 / 判定 |
|---|---|---|
| 输入非法 | 对应槽位失败，其他槽位不受影响 | `INVALID_PARAMETER` / `INVALID_WINDOW` / `INSUFFICIENT_DATA` / `UNKNOWN_*` 等（HTTP 422） |
| 超容量 | **整批拒绝**，不执行任何条目 | `BATCH_SIZE_LIMIT`（HTTP 429） |
| 超时 | 仅未完成槽位失败，已完成结果保留 | `EVALUATION_TIMEOUT`（槽位内；整批 HTTP 仍 200） |
| 内部出错 | 对应槽位失败，原始异常不泄露 | `COMPUTATION_ERROR`（HTTP 500）/ `INTERNAL_ERROR` |

跨版本重算在此之上用判定字段表达：`REPRODUCED` / `CHANGED` /
`REJECTED`（`REJECTED` 槽位的 `error.code` 给出被拒的具体原因，如
`INCOMPATIBLE_VERSION`）。

## 错误码一览

| code | 含义 | HTTP |
|---|---|---|
| `INSUFFICIENT_DATA` | 历史不足/窗口超过样本/非正价格/序列不齐 | 422 |
| `INVALID_WINDOW` | window<1、horizon<1 | 422 |
| `INVALID_PARAMETER` | 置信度越界、冲突入参、空扫描、非法批量配置 | 422 |
| `UNKNOWN_INSTRUMENT` | 标的无行情或未持有 | 422 |
| `UNKNOWN_SCENARIO` | 情景名称未注册 | 422 |
| `UNKNOWN_ASSUMPTION` | 相关性假设名称未注册 | 422 |
| `UNKNOWN_VERSION` | 版本未注册 | 422 |
| `DIMENSION_MISMATCH` | 矩阵维度/标的集合不匹配 | 422 |
| `INVALID_CORRELATION_VALUE` | 相关值越界/非有限/对角非1 | 422 |
| `ASYMMETRIC_MATRIX` | 矩阵非对称 | 422 |
| `NOT_POSITIVE_DEFINITE` | 矩阵非正定 | 422 |
| `INCOMPATIBLE_VERSION` | 跨版本重算未声明兼容且未显式批准 | 422 |
| `BATCH_SIZE_LIMIT` | 批量/扫描/重算超条目上限 | 429 |
| `EVALUATION_TIMEOUT` | 超时间预算（仅未完成槽位） | 槽位内* |
| `COMPUTATION_ERROR` | 内部数值失败（已包装） | 500 |
| `INTERNAL_ERROR` | 未预期错误（兜底，不透出细节） | 500 |

\* 超时发生在某个结果槽位内时，整批/扫描请求本身仍返回 200，仅该槽位
`ok=false` 且 `error.code=EVALUATION_TIMEOUT`；只有在尚未分槽前就触发
的资源错误（如超容量）才整体返回 429。

## 本地验证方法

```bash
# 全量测试（含批量扫描、跨版本三判定、快照/排名确定性、并发时限）
uv run pytest

# 安静模式跑全量（仍会输出每条用例的 CASE START / CASE VERDICT 日志行）
uv run pytest -q

# 只跑本次新增能力
uv run pytest tests/test_scan.py tests/test_recompare.py \
              tests/test_snapshot.py tests/test_scan_api.py -q

# 单模块
uv run pytest tests/test_correlation.py -q
uv run pytest tests/test_batch.py -q
```

> **默认测试命令就带判定依据**：`pytest.ini` 已默认 `log_cli=true`
> （INFO 级），`tests/conftest.py` 用自动夹具为每条用例打印
> `CASE START <用例> | fixtures: <输入摘要>` 与
> `CASE VERDICT <用例> | PASSED/FAILED: <原因>`；各用例内部再以
> `log.info("case=...; criterion: ...")` 打印**输入摘要与判定依据**。
> 直接运行 `uv run pytest` 即可看到，无需再追加
> `-s -o log_cli=true --log-cli-level=INFO` 之类的参数。

测试日志约定：失败时可直接依据"输入摘要 + 判定依据"定位是输入构造问题
还是结论偏离预期。
