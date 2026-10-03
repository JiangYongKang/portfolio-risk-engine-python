# Portfolio Risk Engine（Python / FastAPI）

本地可运行的投资组合风险度量与压力测试服务：多资产敞口聚合、
历史模拟法 + 参数法尾部风险、可命名/可组合压力情景、相关性假设
严格校验、参数集版本化、并发批量评估（顺序稳定、失败隔离），并支持
**组合×情景批量扫描**、**跨版本结论重算对比**与
**确定性快照/最差情形排名**。

- 详细口径、错误码、容量/时限配置与兼容策略：[`docs/README.md`](docs/README.md)
- 入口：`main.py`（FastAPI app）
- 核心代码：`risk_engine/`（series / risk / correlation / scenarios /
  versions / batch / scan / recompare / snapshot / service）
- 测试：`tests/`（89 例，`uv run pytest` 默认即打印每条用例的输入与判定依据）

```bash
uv sync
uv run pytest
uv run uvicorn main:app --reload
```
