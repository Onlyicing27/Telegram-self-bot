"""
Focused tests for the four runtime fixes:
  1. Real Telegram tool context reaches ToolExecutor
  2. Provider success=False triggers fallback chain
  3. Malformed OpenAI tool arguments are detected, not silently {}
  4. MAX_TOOL_ROUNDS does not silently drop pending tool calls
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.ai.providers.base.contract import ProviderResponse
from backend.ai.tools.executor import ToolExecutionResult


# ── Fix 1 ──

def test_attach_tools_preserves_real_context():
    from backend.ai.engine.engine import Engine
    from backend.ai.tools.context import ToolContext
    from backend.ai.tools.registry import ToolRegistry

    fake_telegram = MagicMock(name="TelegramAPI")
    fake_client = MagicMock(name="TelethonClient")
    real_ctx = ToolContext(telegram=fake_telegram, owner_id=999, tz_str="Asia/Tehran", client=fake_client)

    registry = ToolRegistry()
    engine = Engine(tool_registry=registry)
    engine.attach_tools(registry, owner_id=999, tz_str="Asia/Tehran", tool_context=real_ctx)

    assert engine._tool_executor is not None
    base = engine._tool_executor._context
    assert base.telegram is fake_telegram
    assert base.client is fake_client
    assert base.owner_id == 999


def test_attach_tools_without_context_defaults_to_none():
    from backend.ai.engine.engine import Engine
    from backend.ai.tools.registry import ToolRegistry

    registry = ToolRegistry()
    engine = Engine(tool_registry=registry)
    engine.attach_tools(registry, owner_id=1, tz_str="UTC")

    assert engine._tool_executor._context.telegram is None
    assert engine._tool_executor._context.client is None


def test_dispatcher_build_tool_context_inherits_real_telegram():
    from backend.ai.engine.dispatcher import Dispatcher
    from backend.ai.engine.hooks import NOOP_HOOKS
    from backend.ai.engine.metrics import EngineMetrics
    from backend.ai.session.request import AIRequest
    from backend.ai.tools.context import ToolContext
    from backend.ai.tools.executor import ToolExecutor
    from backend.ai.tools.registry import ToolRegistry

    fake_telegram = MagicMock(name="TelegramAPI")
    fake_client = MagicMock(name="TelethonClient")
    base_ctx = ToolContext(telegram=fake_telegram, owner_id=123, tz_str="UTC", client=fake_client)
    registry = ToolRegistry()
    executor = ToolExecutor(registry, base_ctx)

    mock_pm = MagicMock()
    mock_pm.get_active_name.return_value = "test"
    mock_pm.get_active.return_value.config.model = "m"
    mock_pm.chat = AsyncMock(return_value=ProviderResponse(
        text="ok", provider_name="test", success=True,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}))

    mock_conv = MagicMock()
    mock_sess = MagicMock()
    mock_sess.session_id = "s"
    mock_sess.active_provider = "test"
    mock_conv.get_session.return_value = mock_sess
    mock_conv.restore_history = AsyncMock()
    mock_conv.get_history.return_value = []

    mock_pb = MagicMock()
    pp = MagicMock()
    pp.system_prompt = "sys"
    pp.runtime_context = ""
    pp.conversation_context = ""
    pp.tool_context = ""
    pp.user_input = "Hi"
    pp.estimated_tokens.estimated_input_tokens = 10
    pp.estimated_tokens.prompt_size_chars = 20
    mock_pb.build.return_value = pp

    d = Dispatcher(mock_conv, mock_pb, mock_pm, NOOP_HOOKS, EngineMetrics(), tool_executor=executor)
    req = AIRequest(owner_id=123, user_message="Hi", chat_id=456)
    per_req_ctx = d._build_tool_context(req)

    assert per_req_ctx.telegram is fake_telegram
    assert per_req_ctx.client is fake_client
    assert per_req_ctx.owner_id == 123
    assert per_req_ctx.extra.get("chat_id") == 456


# ── Fix 2 ──

@pytest.mark.asyncio
async def test_provider_success_false_triggers_fallback_chain():
    import os
    from backend.ai.providers.manager.manager import ProviderManager
    from backend.ai.providers.registry.registry import ProviderRegistry

    os.environ["AI_PROVIDER_FALLBACK"] = "backup"
    reg = ProviderRegistry()

    primary = MagicMock()
    primary.name = "primary"
    primary.health.return_value = {"healthy": True}
    primary.chat = AsyncMock(return_value=ProviderResponse(
        text="Rate limited", provider_name="primary", success=False, metadata={"http_status": 429}))

    backup = MagicMock()
    backup.name = "backup"
    backup.health.return_value = {"healthy": True}
    backup.chat = AsyncMock(return_value=ProviderResponse(
        text="OK from backup", provider_name="backup", success=True,
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}))

    reg.register(primary)
    reg.register(backup)
    reg.switch_provider("primary")

    pm = ProviderManager(reg)
    pm._load_env_fallback_chain()
    result = await pm.chat([{"role": "user", "content": "hi"}])

    assert result.success is True
    assert result.text == "OK from backup"
    assert primary.chat.call_count == 1
    assert backup.chat.call_count == 1
    os.environ.pop("AI_PROVIDER_FALLBACK", None)


@pytest.mark.asyncio
async def test_all_providers_fail_returns_failure():
    import os
    from backend.ai.providers.manager.manager import ProviderManager
    from backend.ai.providers.registry.registry import ProviderRegistry

    os.environ["AI_PROVIDER_FALLBACK"] = "backup"
    reg = ProviderRegistry()

    primary = MagicMock()
    primary.name = "primary"
    primary.health.return_value = {"healthy": True}
    primary.chat = AsyncMock(return_value=ProviderResponse(
        text="500 error", provider_name="primary", success=False, metadata={"http_status": 500}))

    backup = MagicMock()
    backup.name = "backup"
    backup.health.return_value = {"healthy": True}
    backup.chat = AsyncMock(return_value=ProviderResponse(
        text="quota exceeded", provider_name="backup", success=False, metadata={"http_status": 429}))

    reg.register(primary)
    reg.register(backup)
    reg.switch_provider("primary")

    pm = ProviderManager(reg)
    pm._load_env_fallback_chain()
    result = await pm.chat([{"role": "user", "content": "hi"}])

    assert result.success is False
    os.environ.pop("AI_PROVIDER_FALLBACK", None)


# ── Fix 3 ──

@pytest.mark.asyncio
async def test_openai_compat_valid_arguments_still_parse():
    from backend.ai.providers.base.config import ProviderConfig
    from backend.ai.providers.openai_compat import OpenAICompatProvider

    config = ProviderConfig(api_key="k", enabled=True, base_url="https://api.openai.com/v1", default_model="gpt-4")
    provider = OpenAICompatProvider(config)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "save", "arguments": json.dumps({"type": "forward"})}}]},
            "finish_reason": "tool_calls"}],
        "usage": {}}
    provider._http_client = MagicMock()
    provider._http_client.post = AsyncMock(return_value=mock_resp)

    result = await provider.chat([{"role": "user", "content": "save"}])

    assert result.success is True
    assert result.tool_calls[0]["arguments"] == {"type": "forward"}
    assert result.tool_calls[0].get("_malformed_arguments") is False


@pytest.mark.asyncio
async def test_openai_compat_malformed_arguments_detected():
    from backend.ai.providers.base.config import ProviderConfig
    from backend.ai.providers.openai_compat import OpenAICompatProvider

    config = ProviderConfig(api_key="k", enabled=True, base_url="https://api.openai.com/v1", default_model="gpt-4")
    provider = OpenAICompatProvider(config)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "save", "arguments": "{bad json,,,"}}]},
            "finish_reason": "tool_calls"}],
        "usage": {}}
    provider._http_client = MagicMock()
    provider._http_client.post = AsyncMock(return_value=mock_resp)

    result = await provider.chat([{"role": "user", "content": "save"}])

    assert result.tool_calls[0]["_malformed_arguments"] is True
    assert result.tool_calls[0]["name"] == "save"
    assert result.tool_calls[0]["id"] == "c1"


@pytest.mark.asyncio
async def test_malformed_arguments_not_silently_executed():
    from backend.ai.tools.base import PermissionLevel
    from backend.ai.tools.context import ToolContext
    from backend.ai.tools.executor import ToolExecutor
    from backend.ai.tools.registry import ToolRegistry

    registry = ToolRegistry()
    mock_tool = MagicMock()
    mock_tool.name = "save"
    mock_tool.permission_level = PermissionLevel.READ_ONLY
    mock_tool.execute = AsyncMock(return_value=MagicMock(success=True, message="done", data={}))
    registry.register(mock_tool)

    ctx = ToolContext(telegram=None, owner_id=1, tz_str="UTC")
    executor = ToolExecutor(registry, ctx)

    results = await executor.execute_calls([
        {"id": "c1", "name": "save", "arguments": {}, "_malformed_arguments": True}])

    assert results[0].success is False
    assert results[0].error == "malformed_arguments"
    mock_tool.execute.assert_not_called()


# ── Fix 4 ──

@pytest.mark.asyncio
async def test_tool_round_limit_detected_and_flagged():
    from backend.ai.engine.dispatcher import Dispatcher, MAX_TOOL_ROUNDS
    from backend.ai.engine.hooks import NOOP_HOOKS
    from backend.ai.engine.metrics import EngineMetrics
    from backend.ai.session.request import AIRequest

    mock_pm = MagicMock()
    mock_pm.get_active_name.return_value = "test"
    mock_pm.get_active.return_value.config.model = "m"

    tool_resp = ProviderResponse(text="", provider_name="test", success=True,
        tool_calls=[{"id": "c1", "name": "search", "arguments": {"q": "x"}}],
        usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        metadata={"finish_reason": "tool_calls"})
    mock_pm.chat = AsyncMock(return_value=tool_resp)

    mock_conv = MagicMock()
    mock_sess = MagicMock()
    mock_sess.session_id = "s"
    mock_sess.active_provider = "test"
    mock_conv.get_session.return_value = mock_sess
    mock_conv.restore_history = AsyncMock()
    mock_conv.get_history.return_value = []

    mock_pb = MagicMock()
    pp = MagicMock()
    pp.system_prompt = "sys"
    pp.runtime_context = ""
    pp.conversation_context = ""
    pp.tool_context = ""
    pp.user_input = "Search"
    pp.estimated_tokens.estimated_input_tokens = 50
    pp.estimated_tokens.prompt_size_chars = 100
    mock_pb.build.return_value = pp

    mock_te = MagicMock()
    mock_te.execute_calls = AsyncMock(return_value=[ToolExecutionResult(tool_name="search", success=True, message="ok")])
    mock_te._context = MagicMock()
    mock_te._context.extra = {}

    d = Dispatcher(mock_conv, mock_pb, mock_pm, NOOP_HOOKS, EngineMetrics(), tool_executor=mock_te)
    result = await d.dispatch(AIRequest(owner_id=123, user_message="Search", chat_id=456))

    assert result.success is False
    assert result.metadata.get("tool_round_limit_reached") is True
    assert result.metadata.get("pending_tool_calls") is not None
    assert result.metadata["pending_tool_calls"] > 0
    assert any("maximum" in w for w in result.warnings)
    assert "tool_round_limit_reached" in result.errors
    assert result.metadata.get("tool_rounds") == MAX_TOOL_ROUNDS


@pytest.mark.asyncio
async def test_tool_round_limit_does_not_execute_pending():
    from backend.ai.engine.dispatcher import Dispatcher, MAX_TOOL_ROUNDS
    from backend.ai.engine.hooks import NOOP_HOOKS
    from backend.ai.engine.metrics import EngineMetrics
    from backend.ai.session.request import AIRequest

    mock_pm = MagicMock()
    mock_pm.get_active_name.return_value = "test"
    mock_pm.get_active.return_value.config.model = "m"

    tool_resp = ProviderResponse(text="", provider_name="test", success=True,
        tool_calls=[{"id": "c1", "name": "search", "arguments": {"q": "x"}}],
        usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        metadata={"finish_reason": "tool_calls"})
    mock_pm.chat = AsyncMock(return_value=tool_resp)

    mock_conv = MagicMock()
    mock_sess = MagicMock()
    mock_sess.session_id = "s"
    mock_sess.active_provider = "test"
    mock_conv.get_session.return_value = mock_sess
    mock_conv.restore_history = AsyncMock()
    mock_conv.get_history.return_value = []

    mock_pb = MagicMock()
    pp = MagicMock()
    pp.system_prompt = "sys"
    pp.runtime_context = ""
    pp.conversation_context = ""
    pp.tool_context = ""
    pp.user_input = "Search"
    pp.estimated_tokens.estimated_input_tokens = 50
    pp.estimated_tokens.prompt_size_chars = 100
    mock_pb.build.return_value = pp

    exec_count = 0
    async def fake_execute(*a, **kw):
        nonlocal exec_count
        exec_count += 1
        return [ToolExecutionResult(tool_name="search", success=True, message="ok")]

    mock_te = MagicMock()
    mock_te.execute_calls = AsyncMock(side_effect=fake_execute)
    mock_te._context = MagicMock()
    mock_te._context.extra = {}

    d = Dispatcher(mock_conv, mock_pb, mock_pm, NOOP_HOOKS, EngineMetrics(), tool_executor=mock_te)
    result = await d.dispatch(AIRequest(owner_id=123, user_message="Search", chat_id=456))

    assert exec_count == MAX_TOOL_ROUNDS
    assert result.success is False
