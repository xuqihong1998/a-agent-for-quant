from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from agents.data_agent import DataAgentOutput
from agents.sdk_support import run_structured_agent
from tools.risk_checks import classify_plan_action


VALID_STANCES = {"继续持有", "谨慎持有", "重点观察", "普通观察", "减仓观察", "暂不关注"}


def _normalize_stance(value: Any) -> str:
    mapping = {
        "持有": "继续持有",
        "继续持仓": "继续持有",
        "观察": "普通观察",
    }
    normalized = mapping.get(str(value), str(value))
    return normalized if normalized in VALID_STANCES else "普通观察"


class ResearchOpinion(BaseModel):
    symbol: str
    name: str
    category: Literal["position", "watchlist"]
    theme_view: str
    pattern_view: str
    event_view: str
    candidate_stance: Literal["继续持有", "谨慎持有", "重点观察", "普通观察", "减仓观察", "暂不关注"]
    headline: str
    thesis: str
    key_signals: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    focus_price: str
    next_day_plan: str
    invalidation: str
    confidence: float = Field(ge=0.0, le=1.0)


class ResearchAgentOutput(BaseModel):
    review_date: str
    run_mode: Literal["openai_agent", "rule_fallback"]
    market_summary: str
    portfolio_summary: str
    symbols: list[ResearchOpinion]
    warnings: list[str] = Field(default_factory=list)


def _pattern_view(item: dict[str, Any]) -> str:
    if item["trend"] == "up" and item["amount_ratio_5d"] >= 1.2:
        return "多头结构保持，且量能配合偏积极。"
    if item["trend"] == "up":
        return "趋势仍在多头一侧，但量能延续性还要观察。"
    if item["trend"] == "weak":
        return "价格相对均线转弱，结构更偏防守。"
    return "价格处于方向选择区间，仍需等待进一步确认。"


def _event_view(item: dict[str, Any]) -> str:
    if item["pct_chg"] >= 0.03 and item["amount_ratio_5d"] >= 1.2:
        return "短线触发点来自放量上行，次日重点看强势是否延续。"
    if item["pct_chg"] < 0 and item["amount_ratio_5d"] >= 1.2:
        return "短线触发点来自放量回落，优先关注是否出现资金兑现。"
    if item["amount_ratio_5d"] <= 0.8:
        return "当前更像缩量整理阶段，触发条件要等待量价重新共振。"
    return "当前未见明确事件触发，先按量价结构做跟踪。"


def _theme_view(item: dict[str, Any]) -> str:
    tags = item.get("tags") or []
    if tags:
        return f"当前主题标签集中在 {'/'.join(tags)}。"
    return "当前未配置明确主题标签，按个股自身量价结构跟踪。"


def _build_signals(item: dict[str, Any]) -> list[str]:
    signals = [
        f"收盘 {item['close']:.2f}，MA5/MA10/MA20 为 {item['ma5']:.2f}/{item['ma10']:.2f}/{item['ma20']:.2f}",
        f"量能状态 {item['volume_status']}，5日量比 {item['amount_ratio_5d']:.2f}",
        f"当日涨跌幅 {item['pct_chg'] * 100:.2f}%",
    ]
    if item.get("is_position"):
        signals.append(f"持仓动作基线为 {item.get('position_action', '继续持有')}")
    return signals


def _fallback_symbol(item: dict[str, Any]) -> ResearchOpinion:
    stance = item.get("position_action") if item.get("is_position") else classify_plan_action(item)
    risk_flags = list(item.get("risk_notes", [])) if item.get("risk_notes") else []
    if not risk_flags:
        risk_flags = ["暂无新增风险提示，继续观察量价与均线关系。"]
    return ResearchOpinion(
        symbol=item["symbol"],
        name=item["name"],
        category="position" if item.get("is_position") else "watchlist",
        theme_view=_theme_view(item),
        pattern_view=_pattern_view(item),
        event_view=_event_view(item),
        candidate_stance=_normalize_stance(stance),
        headline=item["comment"],
        thesis=f"{item['comment']}。当前更值得关注的是 {item['next_focus']}。",
        key_signals=_build_signals(item),
        risk_flags=risk_flags,
        focus_price=item["next_focus"],
        next_day_plan=item["action_hint"],
        invalidation="若跌破关键均线并伴随量价转弱，需要下调预期。",
        confidence=0.58,
    )


def build_rule_fallback_research(data_output: DataAgentOutput, warning: str | None = None) -> ResearchAgentOutput:
    positions = [item for item in data_output.symbols if item.is_position]
    warnings = list(data_output.warnings)
    if warning:
        warnings.append(warning)
    portfolio_summary = (
        "持仓侧先看既有强势股能否守住短期均线，弱势持仓则优先控制回撤。"
        if positions
        else "当前无持仓，研究重点放在观察池中潜在走强标的。"
    )
    return ResearchAgentOutput(
        review_date=data_output.review_date,
        run_mode="rule_fallback",
        market_summary="研究结论由规则兜底生成，主要依据趋势、均线、量能与持仓状态。",
        portfolio_summary=portfolio_summary,
        symbols=[_fallback_symbol(item.model_dump()) for item in data_output.symbols],
        warnings=warnings,
    )


def _normalize_research_payload(payload: Any, data_output: DataAgentOutput) -> dict[str, Any]:
    if isinstance(payload, BaseModel):
        payload = payload.model_dump()
    payload = dict(payload or {})
    payload["run_mode"] = "openai_agent"
    payload["review_date"] = str(payload.get("review_date") or data_output.review_date)
    payload["market_summary"] = str(payload.get("market_summary") or "")
    payload["portfolio_summary"] = str(payload.get("portfolio_summary") or "")
    payload["warnings"] = [str(item) for item in (payload.get("warnings") or [])]

    data_map = {item.symbol: item for item in data_output.symbols}
    seen_symbols: set[str] = set()
    normalized_symbols: list[dict[str, Any]] = []
    for raw in payload.get("symbols") or []:
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("symbol", "")).zfill(6)
        source = data_map.get(symbol)
        if source is None:
            continue
        seen_symbols.add(symbol)
        try:
            confidence = float(raw.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        normalized_symbols.append(
            {
                "symbol": symbol,
                "name": str(raw.get("name") or source.name),
                "category": "position" if source.is_position else "watchlist",
                "theme_view": str(raw.get("theme_view") or _theme_view(source.model_dump())),
                "pattern_view": str(raw.get("pattern_view") or _pattern_view(source.model_dump())),
                "event_view": str(raw.get("event_view") or _event_view(source.model_dump())),
                "candidate_stance": _normalize_stance(
                    raw.get("candidate_stance")
                    or source.position_action
                    or classify_plan_action(source.model_dump())
                ),
                "headline": str(raw.get("headline") or source.comment),
                "thesis": str(raw.get("thesis") or source.comment),
                "key_signals": [str(item) for item in (raw.get("key_signals") or _build_signals(source.model_dump()))],
                "risk_flags": [str(item) for item in (raw.get("risk_flags") or source.risk_notes or [])],
                "focus_price": str(raw.get("focus_price") or source.next_focus),
                "next_day_plan": str(raw.get("next_day_plan") or source.action_hint),
                "invalidation": str(raw.get("invalidation") or "若量价结构破坏，则需要下调观点。"),
                "confidence": confidence,
            }
        )
    for symbol, source in data_map.items():
        if symbol in seen_symbols:
            continue
        normalized_symbols.append(_fallback_symbol(source.model_dump()).model_dump())
    payload["symbols"] = normalized_symbols
    return payload


def create_research_agent(data_output: DataAgentOutput, model: str | None = None, structured_output: bool = True) -> Any:
    from agents.sdk_support import import_openai_agents_sdk

    sdk = import_openai_agents_sdk()
    Agent = getattr(sdk, "Agent")
    function_tool = getattr(sdk, "function_tool")
    context_payload = data_output.model_dump()
    symbol_map = {item["symbol"]: item for item in context_payload["symbols"]}

    @function_tool
    def load_data_context() -> dict[str, Any]:
        """Load the deterministic symbol summaries produced by the data agent."""

        return context_payload

    @function_tool
    def get_symbol_summary(symbol: str) -> dict[str, Any]:
        """Get one symbol summary by six-digit code."""

        return symbol_map[str(symbol).zfill(6)]

    instructions = """
你是 A 股研究分析师 research_agent。
你的职责是基于 data_agent 提供的结构化 summary，给出主题、形态、触发因素和候选观点。

严格要求：
1. 先调用 load_data_context 获取全量上下文；必要时再调用 get_symbol_summary。
2. 只能依据给定结构化数据、持仓信息和主题标签做判断。
3. 不能编造新闻、公告、研报、资金流、板块消息。event_view 只能解释为量价或持仓层面的触发因素。
4. 每个标的都必须输出，candidate_stance 只能取给定枚举值。
5. 研究结论要简洁、可审计，重点回答为什么关注、关注什么、什么情况下失效。
"""
    if not structured_output:
        instructions += """
6. 最终输出必须是一个 JSON 对象，不要使用 markdown。
7. 顶层字段必须包含 review_date, run_mode, market_summary, portfolio_summary, symbols, warnings。
8. symbols 中每个元素都必须包含 symbol, name, category, theme_view, pattern_view, event_view, candidate_stance, headline, thesis, key_signals, risk_flags, focus_price, next_day_plan, invalidation, confidence。
"""

    kwargs: dict[str, Any] = {
        "name": "research_agent",
        "instructions": instructions,
        "tools": [load_data_context, get_symbol_summary],
        "model": model,
    }
    if structured_output:
        kwargs["output_type"] = ResearchAgentOutput
    return Agent(**kwargs)


def run_research_agent(
    data_output: DataAgentOutput,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    disable_tracing: bool = False,
    allow_rule_fallback: bool = True,
    prefer_text_json: bool = False,
) -> ResearchAgentOutput:
    if not data_output.symbols:
        return build_rule_fallback_research(data_output, warning="data_agent 未产出可研究标的。")

    prompt = (
        f"请基于 {data_output.review_date} 的 data_agent 输出，生成研究层观点。"
        "输出面向盘后复盘和次日跟踪，不要涉及自动交易执行。"
    )
    try:
        return run_structured_agent(
            create_agent=lambda structured_output: create_research_agent(
                data_output,
                model=model,
                structured_output=structured_output,
            ),
            prompt=prompt,
            output_model=ResearchAgentOutput,
            normalize_output=lambda payload: _normalize_research_payload(payload, data_output),
            api_key=api_key,
            base_url=base_url,
            disable_tracing=disable_tracing,
            prefer_text_json=prefer_text_json,
        )
    except Exception as exc:
        if not allow_rule_fallback:
            raise
        return build_rule_fallback_research(data_output, warning=str(exc))
