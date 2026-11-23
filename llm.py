"""Model adapters. The agent only depends on the LLM protocol below, so tests and
the evaluation's self-check can swap Ollama for a scripted model."""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM(Protocol):
    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMReply: ...


class OllamaLLM:
    def __init__(self, model: str = "llama3.2", temperature: float = 0.0, host: str | None = None) -> None:
        import ollama

        self.model = model
        self.temperature = temperature
        self.client = ollama.AsyncClient(host=host)

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMReply:
        resp = await self.client.chat(
            model=self.model, messages=messages, tools=tools or None,
            options={"temperature": self.temperature},
        )
        msg = resp.message
        calls = []
        for tc in msg.tool_calls or []:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"__unparseable__": args}
            calls.append(ToolCall(tc.function.name, dict(args or {})))
        return LLMReply(
            content=msg.content or "",
            tool_calls=calls,
            prompt_tokens=resp.prompt_eval_count or 0,
            completion_tokens=resp.eval_count or 0,
        )


class ScriptedLLM:
    """Plays back a fixed list of replies. Each item is either an LLMReply or a function
    of the conversation so far that returns one (so a script can read tool results)."""

    def __init__(self, script: list[LLMReply | Callable[[list[dict[str, Any]]], LLMReply]]) -> None:
        self.script = list(script)
        self.calls = 0

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMReply:
        self.calls += 1
        if not self.script:
            return LLMReply("Done.", prompt_tokens=50, completion_tokens=5)
        step = self.script.pop(0)
        return step(messages) if callable(step) else step


def call(name: str, **arguments: Any) -> LLMReply:
    """Shorthand for a scripted reply that calls one tool."""
    return LLMReply("", [ToolCall(name, arguments)], prompt_tokens=400, completion_tokens=30)


def answer(text: str) -> LLMReply:
    return LLMReply(text, prompt_tokens=600, completion_tokens=40)


def last_tool_result(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for m in reversed(messages):
        if m.get("role") == "tool":
            try:
                return json.loads(m["content"])
            except json.JSONDecodeError:
                return {}
    return {}
