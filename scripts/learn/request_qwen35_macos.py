#!/usr/bin/env python3
"""向本地 Qwen3.5 Mac 调试服务发送一条固定的非流式聊天请求。"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL_PATH = "Qwen/Qwen3.5-0.8B"


def parse_args() -> argparse.Namespace:
    """解析本地服务地址和请求超时时间。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:30000",
        help="SGLang HTTP 服务地址，默认 http://127.0.0.1:30000",
    )
    parser.add_argument(
        "--timeout",
        default=120.0,
        type=float,
        help="等待 HTTP 响应的秒数，默认 120",
    )
    return parser.parse_args()


def build_payload() -> dict[str, Any]:
    """构造一条短、确定性的 OpenAI Chat Completions 请求。"""
    return {
        "model": MODEL_PATH,
        "messages": [
            {
                "role": "user",
                "content": "请只回答一个数字：1 + 1 = ?",
            }
        ],
        "temperature": 0,
        "max_tokens": 8,
        "stream": False,
    }


def post_chat_completion(base_url: str, timeout: float) -> dict[str, Any]:
    """调用本地 /v1/chat/completions，并将 HTTP 错误正文输出给调用者。"""
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    request = Request(
        endpoint,
        data=json.dumps(build_payload()).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {endpoint}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"无法连接 {endpoint}: {error.reason}") from error


def assert_nonempty_completion(response: dict[str, Any]) -> None:
    """确认服务返回了至少一个包含文本或 reasoning 内容的 choice。"""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"响应没有 choices：{response}")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"首个 choice 没有 message：{choices[0]}")

    content = message.get("content") or message.get("reasoning_content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"首个 choice 没有非空生成内容：{choices[0]}")


def main() -> int:
    """发送请求、校验结果并打印完整 JSON 响应。"""
    args = parse_args()
    if args.timeout <= 0:
        raise RuntimeError("--timeout 必须大于 0")

    response = post_chat_completion(args.base_url, args.timeout)
    assert_nonempty_completion(response)
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"请求失败：{error}", file=sys.stderr)
        raise SystemExit(1) from error
