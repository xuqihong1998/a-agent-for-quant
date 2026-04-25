from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agents.data_agent import DataAgentOutput
from agents.research_agent import ResearchAgentOutput
from agents.risk_agent import RiskAgentOutput
from agents.sdk_support import import_openai_agents_sdk, run_structured_agent


class PlannedSymbol(BaseModel):
    symbol: str
    name: str
    category: Literal["position", "watchlist"]
    final_action: Literal["继续持有", "谨慎持有", "重点观察", "普通观察", "减仓观察", "暂不关注"]
    headline: str
    thesis: str
    plan_steps: list[str] = Field(default_factory=list)
    focus_price: str
    invalidation: str
    risk_level: Literal["low", "medium", "high"]
    position_pct_limit: float = Field(ge=0.0, le=1.0)
    risk_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class PlanAgentOutput(BaseModel):
    review_date: str
    run_mode: Literal["openai_agent", "rule_fallback"]
    market_summary: str
    portfolio_summary: str
    next_day_overview: str
    symbols: list[PlannedSymbol]
    warnings: list[str] = Field(default_factory=list)


STANCE_RANK = {
    "继续持有": 0,
    "谨慎持有": 1,
    "重点观察": 2,
    "普通观察": 3,
    "减仓观察": 4,
    "暂不关注": 5,
}


def _clamp_final_action(candidate: str, upper_bound: str) -> str:
    candidate_rank = STANCE_RANK.get(str(candidate), STANCE_RANK["普通观察"])
    upper_bound_rank = STANCE_RANK.get(str(upper_bound), STANCE_RANK["普通观察"])
    return candidate if candidate_rank >= upper_bound_rank else upper_bound


def _fallback_plan_steps(data_item: dict[str, Any], risk_item: dict[str, Any]) -> list[str]:
    steps = [str(data_item.get("action_hint") or "按既有规则继续跟踪。")]
    if risk_item.get("risk_level") == "high":
        steps.append("开盘后优先确认风险位是否继续失守，必要时先降级处理。")
    elif risk_item.get("risk_level") == "medium":
        steps.append("先看关键均线附近承接，不满足条件不升级动作。")
    else:
        steps.append("若量价配合延续，可按计划继续观察或持有。")
    return steps


def build_rule_fallback_plan(
    data_output: DataAgentOutput,
    research_output: ResearchAgentOutput,
    risk_output: RiskAgentOutput,
    warning: str | None = None,
) -> PlanAgentOutput:
    data_map = {item.symbol: item for item in data_output.symbols}
    research_map = {item.symbol: item for item in research_output.symbols}
    symbols: list[PlannedSymbol] = []

    for risk in risk_output.symbols:
        data_item = data_map[risk.symbol]
        research_item = research_map.get(risk.symbol)
        symbols.append(
            PlannedSymbol(
                symbol=risk.symbol,
                name=risk.name,
                category=risk.category,
                final_action=risk.approved_stance,
                headline=research_item.headline if research_item else data_item.comment,
                thesis=research_item.thesis if research_item else data_item.comment,
                plan_steps=_fallback_plan_steps(data_item.model_dump(), risk.model_dump()),
                focus_price=research_item.focus_price if research_item else data_item.next_focus,
                invalidation=research_item.invalidation if research_item else "若量价结构明显转弱，则取消原计划。",
                risk_level=risk.risk_level,
                position_pct_limit=risk.position_pct_limit,
                risk_flags=list(risk.risk_flags),
                confidence=min((research_item.confidence if research_item else 0.55), risk.confidence),
            )
        )

    warnings = list(risk_output.warnings)
    if warning:
        warnings.append(warning)
    return PlanAgentOutput(
        review_date=data_output.review_date,
        run_mode="rule_fallback",
        market_summary=research_output.market_summary,
        portfolio_summary=risk_output.portfolio_risk_summary,
        next_day_overview="次日先按风险等级排序处理持仓，再观察候选标的是否满足计划条件。",
        symbols=symbols,
        warnings=warnings,
    )


def _normalize_plan_payload(
    payload: Any,
    data_output: DataAgentOutput,
    research_output: ResearchAgentOutput,
    risk_output: RiskAgentOutput,
) -> dict[str, Any]:
    if isinstance(payload, BaseModel):
        payload = payload.model_dump()
    payload = dict(payload or {})
    payload["review_date"] = str(payload.get("review_date") or data_output.review_date)
    payload["run_mode"] = "openai_agent"
    payload["market_summary"] = str(payload.get("market_summary") or research_output.market_summary)
    payload["portfolio_summary"] = str(payload.get("portfolio_summary") or risk_output.portfolio_risk_summary)
    payload["next_day_overview"] = str(payload.get("next_day_overview") or "")
    payload["warnings"] = [str(item) for item in (payload.get("warnings") or [])]

    data_map = {item.symbol: item for item in data_output.symbols}
    research_map = {item.symbol: item for item in research_output.symbols}
    risk_map = {item.symbol: item for item in risk_output.symbols}
    seen_symbols: set[str] = set()
    normalized_symbols: list[dict[str, Any]] = []
    for raw in payload.get("symbols") or []:
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("symbol", "")).zfill(6)
        data_item = data_map.get(symbol)
        risk_item = risk_map.get(symbol)
        if data_item is None or risk_item is None:
            continue
        seen_symbols.add(symbol)
        research_item = research_map.get(symbol)
        try:
            position_pct_limit = float(raw.get("position_pct_limit", risk_item.position_pct_limit))
        except (TypeError, ValueError):
            position_pct_limit = risk_item.position_pct_limit
        try:
            confidence = float(raw.get("confidence", min(risk_item.confidence, research_item.confidence if research_item else 0.55)))
        except (TypeError, ValueError):
            confidence = min(risk_item.confidence, research_item.confidence if research_item else 0.55)
        normalized_symbols.append(
            {
                "symbol": symbol,
                "name": str(raw.get("name") or data_item.name),
                "category": "position" if data_item.is_position else "watchlist",
                "final_action": _clamp_final_action(str(raw.get("final_action") or risk_item.approved_stance), risk_item.approved_stance),
                "headline": str(raw.get("headline") or (research_item.headline if research_item else data_item.comment)),
                "thesis": str(raw.get("thesis") or (research_item.thesis if research_item else data_item.comment)),
                "plan_steps": [str(item) for item in (raw.get("plan_steps") or _fallback_plan_steps(data_item.model_dump(), risk_item.model_dump()))],
                "focus_price": str(raw.get("focus_price") or (research_item.focus_price if research_item else data_item.next_focus)),
                "invalidation": str(raw.get("invalidation") or (research_item.invalidation if research_item else "若量价结构转弱，则取消原计划。")),
                "risk_level": risk_item.risk_level,
                "position_pct_limit": position_pct_limit,
                "risk_flags": [str(item) for item in (raw.get("risk_flags") or list(risk_item.risk_flags))],
                "confidence": confidence,
            }
        )
    for symbol, risk_item in risk_map.items():
        if symbol in seen_symbols:
            continue
        data_item = data_map[symbol]
        research_item = research_map.get(symbol)
        normalized_symbols.append(
            PlannedSymbol(
                symbol=symbol,
                name=data_item.name,
                category="position" if data_item.is_position else "watchlist",
                final_action=risk_item.approved_stance,
                headline=research_item.headline if research_item else data_item.comment,
                thesis=research_item.thesis if research_item else data_item.comment,
                plan_steps=_fallback_plan_steps(data_item.model_dump(), risk_item.model_dump()),
                focus_price=research_item.focus_price if research_item else data_item.next_focus,
                invalidation=research_item.invalidation if research_item else "若量价结构转弱，则取消原计划。",
                risk_level=risk_item.risk_level,
                position_pct_limit=risk_item.position_pct_limit,
                risk_flags=list(risk_item.risk_flags),
                confidence=min(risk_item.confidence, research_item.confidence if research_item else 0.55),
            ).model_dump()
        )
    payload["symbols"] = normalized_symbols
    return payload


def create_plan_agent(
    data_output: DataAgentOutput,
    research_output: ResearchAgentOutput,
    risk_output: RiskAgentOutput,
    model: str | None = None,
    structured_output: bool = True,
) -> Any:
    sdk = import_openai_agents_sdk()
    Agent = getattr(sdk, "Agent")
    function_tool = getattr(sdk, "function_tool")
    payload = {
        "data_agent": data_output.model_dump(),
        "research_agent": research_output.model_dump(),
        "risk_agent": risk_output.model_dump(),
    }

    @function_tool
    def load_pipeline_context() -> dict[str, Any]:
        """Load the data, research, and risk stage outputs for planning."""

        return payload

    instructions = """
你是 A 股次日计划汇总员 plan_agent。
你的任务是整合 data_agent、research_agent、risk_agent 的输出，生成次日执行前计划。

严格要求：
1. 必须以 risk_agent 的 approved_stance 作为最终动作上限，不能比它更激进。
2. 只能基于已有结构化输入做总结，不要引入外部新闻或盘中假设。
3. 每个标的都必须输出 final_action、plan_steps、focus_price、invalidation。
4. 持仓优先写风险处理与观察条件，观察池优先写触发条件与失效条件。
5. 输出面向人工复盘和次日开盘前准备，不涉及自动下单。
"""
    if not structured_output:
        instructions += """
6. 最终输出必须是一个 JSON 对象，不要使用 markdown。
7. 顶层字段必须包含 review_date, run_mode, market_summary, portfolio_summary, next_day_overview, symbols, warnings。
8. symbols 中每个元素都必须包含 symbol, name, category, final_action, headline, thesis, plan_steps, focus_price, invalidation, risk_level, position_pct_limit, risk_flags, confidence。
"""

    kwargs: dict[str, Any] = {
        "name": "plan_agent",
        "instructions": instructions,
        "tools": [load_pipeline_context],
        "model": model,
    }
    if structured_output:
        kwargs["output_type"] = PlanAgentOutput
    return Agent(**kwargs)


def run_plan_agent(
    data_output: DataAgentOutput,
    research_output: ResearchAgentOutput,
    risk_output: RiskAgentOutput,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    disable_tracing: bool = False,
    allow_rule_fallback: bool = True,
    prefer_text_json: bool = False,
) -> PlanAgentOutput:
    if not risk_output.symbols:
        return build_rule_fallback_plan(data_output, research_output, risk_output, warning="risk_agent 未产出可规划标的。")

    prompt = (
        f"请为 {data_output.review_date} 整理次日交易计划。"
        "计划应遵守 risk_agent 的动作上限，并面向人工盘前准备。"
    )
    try:
        return run_structured_agent(
            create_agent=lambda structured_output: create_plan_agent(
                data_output,
                research_output,
                risk_output,
                model=model,
                structured_output=structured_output,
            ),
            prompt=prompt,
            output_model=PlanAgentOutput,
            normalize_output=lambda payload: _normalize_plan_payload(payload, data_output, research_output, risk_output),
            api_key=api_key,
            base_url=base_url,
            disable_tracing=disable_tracing,
            prefer_text_json=prefer_text_json,
        )
    except Exception as exc:
        if not allow_rule_fallback:
            raise
        return build_rule_fallback_plan(data_output, research_output, risk_output, warning=str(exc))
