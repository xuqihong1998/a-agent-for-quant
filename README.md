# A-Share Agent Quant

一个面向 A 股量化研究的 AI Agent 项目。  
第一版目标不是自动下单，而是做出一套可运行的：

- 盘后复盘
- 基础指标计算
- 简单策略回测
- 风险检查
- 次日交易计划生成

## 1. 项目目标

本项目希望通过 AI Agent + 量化研究的方式，逐步搭建一套 A 股量化辅助系统。

第一阶段聚焦：

- 拉取 A 股日线数据
- 维护自选股与持仓
- 计算基础技术指标
- 输出盘后复盘报告
- 生成次日观察计划

后续阶段扩展：

- 因子研究
- 策略回测
- 风控 Agent
- 工作流编排
- 半自动交易接口

---

## 2. 当前 MVP 范围

### 已规划功能
- [ ] 从 AKShare 拉取 A 股日线数据
- [ ] 保存本地缓存（CSV / Parquet / SQLite）
- [ ] 读取自选股列表
- [ ] 计算 MA5 / MA10 / MA20 / ATR14
- [ ] 生成盘后 Markdown 报告
- [ ] 生成次日观察清单

### 暂不做
- [ ] 自动下单
- [ ] 实时 tick 策略
- [ ] 高频交易
- [ ] 多 Agent 自由协作
- [ ] 复杂机器学习选股

---

## 3. 技术栈

### 核心
- Python 3.11
- pandas
- akshare
- pyarrow
- pydantic
- openai-agents-python（第 2 阶段接入）
- backtrader（第 2 阶段接入）
- langgraph（第 3 阶段接入）

### 可选
- FastAPI：后续 API 服务
- SQLite：轻量本地存储
- Parquet：行情数据缓存
- TypeScript + Next.js：后续可视化面板
- Go：后续任务调度与服务层
- C++：后续高性能工具层

---

## 4. 项目结构

```text
a_share_agent_quant/
├─ agents/
│  ├─ data_agent.py
│  ├─ research_agent.py
│  ├─ risk_agent.py
│  ├─ plan_agent.py
│  └─ graph.py
├─ tools/
│  ├─ market_data.py
│  ├─ indicators.py
│  ├─ portfolio.py
│  ├─ risk_checks.py
│  ├─ report_writer.py
│  └─ backtest.py
├─ strategies/
│  ├─ breakout.py
│  └─ factor_score.py
├─ scripts/
│  ├─ sync_market_data.py
│  ├─ run_daily_review.py
│  ├─ generate_plan.py
│  └─ run_backtest.py
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ cache/
├─ config/
│  ├─ universe.yaml
│  ├─ positions.yaml
│  ├─ risk_rules.yaml
│  └─ settings.yaml
├─ reports/
│  ├─ daily/
│  └─ backtests/
├─ tests/
├─ .env.example
├─ requirements.txt
└─ README.md