# Portfolio Risk Engine

本地可运行的投资组合风险度量与压力测试服务。目标：在不同风险口径、相关性
假设与版本变更下，结论**一致、可复现、可解释**。

- 多资产敞口聚合，同时给出**历史模拟法**与**参数法（正态假设）**两类 VaR / ES
- 可命名、可组合的压力情景，情景结论与逐因子扰动严格自洽
- 相关性/协方差矩阵严格校验：维度、对称性、正定性、取值范围逐因拒绝，绝不静默修正
- 参数集与情景版本化：旧版本在新规则下重算可复现，或显式标记不兼容后才改变
- 并发批量评估：与串行结果一致、输出顺序稳定、单项失败相互隔离，超限/超时可区分
- 所有领域错误都带稳定的 `reason` 代码，底层数值异常不外泄

## 快速开始

```bash
uv sync                       # 安装依赖（fastapi / numpy / pytest / httpx）
uv run uvicorn main:app --port 8000
# 交互文档: http://127.0.0.1:8000/docs
```

本地验证：

```bash
uv run python -m pytest -q                 # 全部单测（含输入摘要/判定依据日志）
uv run python -m pytest tests -q -o log_cli=true --log-cli-level=INFO
```

冒烟脚本（启动服务后另开终端）：

```bash
curl -s localhost:8000/health
```

## 数据模型与 HTTP 接口

| 方法 & 路径 | 说明 |
| --- | --- |
| `POST /market/prices` | 载入某标的的价格序列（时间正序，至少 2 个正的有限价格） |
| `POST /risk` | 组合敞口聚合 + 历史/参数双口径 VaR、ES；可选 `correlation_override` 替换参数法相关性 |
| `POST /scenarios` | 定义命名情景（`factor_shock`）或组合情景（`composite`） |
| `POST /scenarios/{name}/evaluate` | 对组合施加情景，返回总盈亏与逐因子贡献 |
| `POST /correlation/validate` | 严格校验相关性矩阵（亦可传协方差，库内支持 `kind`） |
| `POST /batch` | 并发批量风险评估，顺序稳定、失败隔离、超时可控 |
| `POST /parameters` | 注册带版本号与规则代号的参数集（同时冻结重放快照） |
| `POST /parameters/{name}/versions/{v}/evaluate` | 按版本重算历史结论 |

错误响应统一为：

```json
{"error": {"reason": "symbol_not_found", "detail": "no market data for symbol 'X'"}}
```

请求体本身不合法（类型/取值范围）返回 422 `invalid_request` 并附字段级
`issues`；任何未预期的服务端异常记录到日志后返回 500 `internal_error`，
不向前端泄漏堆栈或底层数值实现信息。

## 风险指标口径

### 敞口聚合

- 持仓 `quantity`（可为负）乘以该标的**最新价**得到市值；同一标的多笔合并。
- 组合总市值必须有限且为正（不支持净空头组合的权重归一化），否则
  `invalid_position`。
- 权重 `w_i = 市值_i / 组合总市值`。

### 收益与窗口

- 简单收益 `r_t = P_t / P_{t-1} - 1`；取最近 `window` 个收益观测，所有标的
  使用**同一窗口长度**。
- `window` 必须是整数且 `>= 2`；可用观测不足时返回 `insufficient_history`
  （与结构性非法窗口 `invalid_window` 明确区分）。
- 置信度 `confidence ∈ (0.5, 1)`，表示**损失**分位点；报告的 VaR 为非负损失金额。

### 历史模拟法

- 组合历史收益 `R_t = Σ_i w_i · r_{i,t}`，经验损失 `L_t = -R_t`。
- `VaR_h = quantile(L, confidence)`（numpy 线性插值，确定性）。
- `ES_h = mean({ L_t : L_t >= VaR_h })`。
- 由构造有 `ES_h >= VaR_h >= 0`。

### 参数法（多元正态）

同一窗口上估计样本均值 `μ` 与样本协方差（`ddof=1`），令
`z = Φ⁻¹(confidence)`、`σ_p = sqrt(w' Σ w)`：

- `VaR_p = z·σ_p − μ_p`（含漂移调整，下限 0）
- `ES_p  = σ_p·φ(z)/(1−confidence) − μ_p`（正态尾部条件期望）

`Φ⁻¹` 使用 Acklam 系数的确定性实现，不依赖随机数，因此同输入逐字节复现。
零波动序列（全部价格相同）会得到 VaR = ES = 0，不会产生 NaN。

## 压力情景与相关性假设

### 情景

- `FactorShock(factor, shock)`：对命名因子的**相对**冲击，如 `equity=-0.10`
  表示权益敏感部分下跌 10%。
- 标的到因子的映射 `{标的: {因子: 载荷}}`；未显式映射的标的默认 1:1 映射到
  同名因子。
- 标的冲击收益 `r_i = Σ_f 载荷_if · shock_f`。
- 组合情景（`composite`）把各组件情景的同因子冲击**相加**；组件必须都是
  `factor_shock`（禁止嵌套组合），禁止自引用与重复因子。
- **自洽性**：`factor_contributions[f]` 用与总盈亏相同的载荷、仅扰动单因子
  计算，因此 `Σ_f 贡献_f = 总盈亏`（引擎内部还做了恒等校验，违例直接拒绝）。
- 情景中出现任何无法映射到持仓的因子，返回 `scenario_definition_invalid`。

`POST /risk` 的可选字段 `correlation_override`（N×N，N 为组合中不同标的数，
顺序按持仓首次出现的去重顺序）只替换**参数法**所用的经验相关性（窗口波动率
保留），历史模拟法因重放真实联合历史而不变；矩阵按同样的严格规则校验，维度
不符返回 `matrix_dimension_mismatch`。

### 相关性 / 协方差矩阵

严格校验、绝不静默修正（不做最近正定矩阵投影等）。拒绝原因可区分：

| reason | 触发条件 |
| --- | --- |
| `matrix_dimension_mismatch` | 非二维/非方阵/行列不等长/标签数与维度不符 |
| `matrix_invalid_value` | NaN/Inf、非数值、相关系数越界 [-1,1]、对角线不为 1、协方差对角非正 |
| `matrix_asymmetric` | `max|A−A'|` 超过容差（相对 1e-10） |
| `matrix_not_positive_definite` | 最小特征值不大于 `1e-10 × 谱尺度`（含半正定奇异阵） |

`correlation_to_covariance(C, σ) = diag(σ)·C·diag(σ)`，波动率须有限为正。
numpy 的 `LinAlgError` 等底层异常会被翻译为上述领域错误，不透出调用栈。

## 版本兼容策略

- 参数集含 `version`（自身版本）、`rule_version`（编写时的计算规则代号）、
  `compatible_with`（声明仍可复现的旧/新规则代号列表）。
- 注册时冻结一份**重放快照**（情景名、组合、因子映射）；`(name, version)`
  不可变，重复注册返回 `version_incompatible`。
- 在当前规则下重算历史结论：
  - 当前规则 == 编写规则，或出现在 `compatible_with` 中 →
    `reproducible=true`，按固定规则代号重放快照，结论可复现；
  - 否则默认拒绝（`version_incompatible`，HTTP 409）；调用方显式传
    `accept_incompatible=true` 后才按新规则计算，结果标记
    `reproducible=false` 并在 `note` 中写明规则变更。
- 参数集/版本不存在 → `version_not_found`（404），与不兼容错误可区分。

## 并发批量评估

- 线程池并发执行彼此独立的无状态计算；**纯数值路径与串行完全相同**
  （测试以串行参考实现逐项比对）。
- 输出严格按**输入顺序**返回，与完成先后无关。
- 单项失败捕获为 `{ok:false, reason, detail}`，其余项照常完成；任何未预期
  异常统一归为 `internal_error`，不泄漏实现细节。
- 请求数超过 `max_items` → 整体前置拒绝 `batch_too_large`（429）；重复 id、
  非法参数 → `batch_rejected`（422）。
- 超过 `timeout` 墙钟预算的未完成项标记为 `batch_timeout`（408），接口仍按序
  快速返回已完成项与超时项。

## 错误码总览

`symbol_not_found` / `insufficient_history` / `invalid_window` /
`empty_portfolio` / `invalid_position` / `invalid_price_series` /
`matrix_dimension_mismatch` / `matrix_asymmetric` /
`matrix_not_positive_definite` / `matrix_invalid_value` /
`scenario_not_found` / `scenario_definition_invalid` /
`version_not_found` / `version_incompatible` /
`batch_too_large` / `batch_timeout` / `batch_rejected` / `internal_error`

## 代码布局

```
app/
  errors.py            # 稳定 reason 代码的领域异常体系
  models.py            # pydantic 数据模型
  market_data.py       # 本地行情存储（内存）
  risk/metrics.py      # 收益、历史模拟/参数法 VaR、ES
  risk/covariance.py   # 相关性/协方差严格校验与换算
  risk/scenarios.py    # 命名/组合压力情景
  versioning.py        # 版本注册表与可复现重放
  batch.py             # 并发批量、顺序、隔离、超时
  service.py           # 装配门面
  main.py              # FastAPI 路由与错误信封
tests/                 # 矩阵非法输入、边界/极端情景、版本、并发批量、HTTP
```
