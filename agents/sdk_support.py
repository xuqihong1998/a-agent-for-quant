from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _log(message: str) -> None:
    print(f"[SDK] {message}")


def import_openai_agents_sdk() -> Any:
    original_package = sys.modules.get("agents")
    removed_paths: list[tuple[int, str]] = []

    for index in range(len(sys.path) - 1, -1, -1):
        candidate = sys.path[index]
        resolved = Path(candidate or ".").resolve()
        if resolved == PROJECT_ROOT:
            removed_paths.append((index, candidate))
            sys.path.pop(index)

    if original_package is not None:
        package_file = getattr(original_package, "__file__", None)
        package_path = Path(package_file).resolve() if package_file else None
        package_paths = [Path(path).resolve() for path in getattr(original_package, "__path__", [])]
        is_local_package = bool(package_path and PROJECT_ROOT in package_path.parents) or any(
            PROJECT_ROOT in path.parents or path == PROJECT_ROOT / "agents" for path in package_paths
        )
        if is_local_package:
            del sys.modules["agents"]

    try:
        return importlib.import_module("agents")
    finally:
        for index, candidate in sorted(removed_paths, key=lambda item: item[0]):
            sys.path.insert(index, candidate)
        if original_package is not None and "agents" not in sys.modules:
            sys.modules["agents"] = original_package


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def is_json_mode_unsupported_error(exc: Exception) -> bool:
    return "json mode is not supported" in str(exc).lower()


def build_run_config(
    sdk: Any,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    RunConfig = getattr(sdk, "RunConfig", None)
    MultiProvider = getattr(sdk, "MultiProvider", None)
    if not (RunConfig and MultiProvider and (api_key or base_url)):
        return None

    provider_kwargs: dict[str, Any] = {
        "openai_prefix_mode": "model_id",
        "unknown_prefix_mode": "model_id",
    }
    if api_key:
        provider_kwargs["openai_api_key"] = api_key
    if base_url:
        provider_kwargs["openai_base_url"] = base_url
    return RunConfig(model_provider=MultiProvider(**provider_kwargs))


def run_structured_agent(
    *,
    create_agent: Callable[[bool], Any],
    prompt: str,
    output_model: Any,
    normalize_output: Callable[[Any], Any] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    disable_tracing: bool = False,
    prefer_text_json: bool = False,
) -> Any:
    sdk = import_openai_agents_sdk()
    _log(
        "run_structured_agent start "
        f"prefer_text_json={prefer_text_json} base_url={'set' if base_url else 'default'} "
        f"api_key={'set' if api_key else 'default'}"
    )
    if disable_tracing and hasattr(sdk, "set_tracing_disabled"):
        sdk.set_tracing_disabled(True)
        _log("tracing disabled")

    Runner = getattr(sdk, "Runner")
    run_config = build_run_config(sdk, api_key=api_key, base_url=base_url)
    _log(f"run_config={'configured' if run_config is not None else 'default'}")

    use_structured_output = not prefer_text_json
    _log(f"first_attempt structured_output={use_structured_output}")

    try:
        if run_config is not None:
            result = Runner.run_sync(create_agent(use_structured_output), prompt, run_config=run_config)
        else:
            result = Runner.run_sync(create_agent(use_structured_output), prompt)
    except Exception as exc:
        _log(f"first_attempt exception={type(exc).__name__}: {exc}")
        if prefer_text_json or not is_json_mode_unsupported_error(exc):
            raise
        _log("retrying with structured_output=False after json mode unsupported")
        if run_config is not None:
            result = Runner.run_sync(create_agent(False), prompt, run_config=run_config)
        else:
            result = Runner.run_sync(create_agent(False), prompt)

    raw_output = getattr(result, "final_output", None)
    _log(f"raw_output_type={type(raw_output).__name__}")
    if isinstance(raw_output, str):
        _log(f"raw_output_length={len(raw_output)}")
        payload = extract_json_object(raw_output)
    elif hasattr(raw_output, "model_dump"):
        payload = raw_output.model_dump()
    else:
        payload = raw_output

    if normalize_output is not None:
        _log("normalizing output payload")
        payload = normalize_output(payload)
    validated = output_model.model_validate(payload)
    _log(f"output_model_validated={getattr(output_model, '__name__', str(output_model))}")
    return validated
