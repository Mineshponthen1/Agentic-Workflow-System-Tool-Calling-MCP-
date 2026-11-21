import json
from typing import Any
import ollama
from tools import TOOLS


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "database_query",
            "description": "Look up customer or account status and balance by account_id (e.g. ACC-101, ACC-102).",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "simulate_timeout": {"type": "boolean"},
                },
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "document_search",
            "description": "Search policy or operational documents by query and optional category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string", "enum": ["all", "compliance", "hr", "operations"]},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_lookup",
            "description": "Check available time slots for a team calendar on a YYYY-MM-DD date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "calendar_name": {"type": "string"},
                    "date_iso": {"type": "string"},
                },
                "required": ["calendar_name", "date_iso"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "email_draft",
            "description": "Create a draft email (does not send email).",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ticket_creation",
            "description": "Create a support or compliance ticket with priority low, medium, or high.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                    "description": {"type": "string"},
                    "account_id": {"type": "string"},
                },
                "required": ["title", "priority", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "status_lookup",
            "description": "Check current status of an existing ticket ID (e.g. TKT-101).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string"},
                },
                "required": ["ticket_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "threshold_check",
            "description": "Compare a numeric amount against a threshold in code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "threshold": {"type": "number"},
                    "label": {"type": "string"},
                },
                "required": ["amount", "threshold"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "audit_log_lookup",
            "description": "Look up recorded audit events for an entity_id (e.g. ACC-102).",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                },
                "required": ["entity_id"],
            },
        },
    },
]


def execute_tool_safely(tool_name: str, raw_args: Any) -> dict:
    """Executes a registered tool while handling unknown tools and non-dict arguments."""
    func = TOOLS.get(tool_name)
    if func is None:
        return {
            "error": f"Unknown tool '{tool_name}'. Available tools: {', '.join(TOOLS.keys())}"
        }

    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError:
            return {"error": f"Malformed JSON arguments for tool '{tool_name}'."}

    if not isinstance(raw_args, dict):
        return {"error": f"Arguments for tool '{tool_name}' must be a JSON object."}

    return func(raw_args)


def run_agent_step(
    user_prompt: str,
    model: str = "llama3.2",
    max_tool_calls: int = 4,
    chat_fn=ollama.chat,
) -> dict:
    """Runs a tool-calling loop with Ollama and records tool execution results."""
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a banking operations assistant. Use the provided tools when needed "
                "and give a clear, concise final answer based only on tool outputs."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    tool_events: list[dict[str, Any]] = []

    try:
        for _ in range(max_tool_calls):
            response = chat_fn(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                options={"temperature": 0},
            )
            msg = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", None)
            if msg is None:
                break

            tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
            content = msg.get("content", "") if isinstance(msg, dict) else (getattr(msg, "content", "") or "")

            if not tool_calls:
                return {
                    "final_answer": content,
                    "tool_events": tool_events,
                    "status": "completed",
                }

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            )

            for call in tool_calls:
                fn = call.get("function", {}) if isinstance(call, dict) else getattr(call, "function", {})
                name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
                args = fn.get("arguments", {}) if isinstance(fn, dict) else getattr(fn, "arguments", {})

                result = execute_tool_safely(name, args)
                tool_events.append({"tool": name, "args": args, "result": result})

                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result),
                    }
                )

        # Final summary pass after tool calls
        final_resp = chat_fn(
            model=model,
            messages=messages,
            options={"temperature": 0},
        )
        final_msg = final_resp.get("message", {}) if isinstance(final_resp, dict) else getattr(final_resp, "message", {})
        final_text = final_msg.get("content", "") if isinstance(final_msg, dict) else (getattr(final_msg, "content", "") or "")
        return {
            "final_answer": final_text,
            "tool_events": tool_events,
            "status": "completed",
        }
    except Exception as exc:
        return {
            "final_answer": "",
            "tool_events": tool_events,
            "status": "error",
            "error": str(exc),
        }


if __name__ == "__main__":
    out = run_agent_step("Check account ACC-102 and see if 25000 exceeds threshold 10000.")
    print(json.dumps(out, indent=2))