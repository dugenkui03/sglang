#!/usr/bin/env python3
"""Run a four-call tool-use conversation and verify Radix KV-cache reuse."""

import argparse
import json
import time
import urllib.error
import urllib.request


def request_json(url, payload=None, attempts=10):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == attempts:
                raise
            print(f"request attempt {attempt} failed: {exc}; retrying...", flush=True)
            time.sleep(min(attempt, 5))


def wait_until_healthy(base_url, attempts=60):
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        if attempt == attempts:
            raise RuntimeError(f"server did not become healthy: {base_url}")
        time.sleep(2)


def cached_tokens(response):
    details = response.get("usage", {}).get("prompt_tokens_details") or {}
    return details.get("cached_tokens", 0)


def complete(base_url, messages, tools, tool_choice, max_tokens=96):
    return request_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": "qwen38-27b",
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "return_cached_tokens_details": True,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    wait_until_healthy(base_url)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Add two integers and return their sum.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a deterministic arithmetic agent. Use the supplied tool "
                "when asked to add numbers. After receiving a tool result, answer "
                "with only the final integer. Keep the full conversation history."
            ),
        },
        {"role": "user", "content": "Use add_numbers to calculate 17 + 25."},
    ]

    first = complete(base_url, messages, tools, "auto")
    assistant = first["choices"][0]["message"]
    calls = assistant.get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != "add_numbers":
        raise RuntimeError(f"expected one add_numbers tool call, got: {assistant}")
    call = calls[0]
    arguments = json.loads(call["function"]["arguments"])
    result = arguments["a"] + arguments["b"]
    messages.extend(
        [
            assistant,
            {"role": "tool", "tool_call_id": call["id"], "content": str(result)},
        ]
    )

    second = complete(base_url, messages, tools, "auto", max_tokens=32)
    answer = second["choices"][0]["message"]
    messages.extend([answer, {"role": "user", "content": "Now add 8 to that result."}])
    third = complete(base_url, messages, tools, "auto")
    next_assistant = third["choices"][0]["message"]
    next_calls = next_assistant.get("tool_calls") or []
    if len(next_calls) != 1 or next_calls[0]["function"]["name"] != "add_numbers":
        raise RuntimeError(
            f"expected a second add_numbers tool call, got: {next_assistant}"
        )
    next_call = next_calls[0]
    next_arguments = json.loads(next_call["function"]["arguments"])
    next_result = next_arguments["a"] + next_arguments["b"]
    messages.extend(
        [
            next_assistant,
            {
                "role": "tool",
                "tool_call_id": next_call["id"],
                "content": str(next_result),
            },
        ]
    )
    fourth = complete(base_url, messages, tools, "auto", max_tokens=32)

    cache_counts = [cached_tokens(item) for item in (first, second, third, fourth)]
    print("tool arguments:", arguments)
    print("tool result:", result)
    print("turn 2 answer:", answer.get("content"))
    print("turn 3 tool arguments:", next_arguments)
    print("turn 3 tool result:", next_result)
    print("turn 4 answer:", fourth["choices"][0]["message"].get("content"))
    print("cached tokens by call:", cache_counts)
    if any(count <= 0 for count in cache_counts[1:]):
        raise RuntimeError(f"KV cache was not reused: {cache_counts}")
    print("agent multi-turn and KV-cache verification passed")


if __name__ == "__main__":
    main()
