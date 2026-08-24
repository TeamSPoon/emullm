"""Interactive command-line client for an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8801/v1"
DEFAULT_MODEL = "yourself/same"


class ChatError(RuntimeError):
    """A useful error returned by the chat service or HTTP client."""


def _chat_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        if parts:
            return "".join(parts)
    raise ChatError("The server response did not contain textual assistant content.")


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 900.0,
    ) -> None:
        self.url = _chat_url(base_url)
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps({"model": self.model, "messages": messages}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.url, data=payload, headers=headers, method="POST")

        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                body = json.load(response)
        except HTTPError as exc:
            detail = exc.reason
            try:
                error_body = json.loads(exc.read().decode("utf-8"))
                detail = error_body.get("detail") or error_body.get("error", {}).get("message") or detail
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                pass
            raise ChatError(f"Server returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise ChatError(f"Could not connect to {self.url}: {exc.reason}") from exc
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ChatError("The server returned an invalid chat-completion response.") from exc

        try:
            return _message_text(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ChatError("The server returned an invalid chat-completion response.") from exc


class Conversation:
    def __init__(self, system_prompt: str | None = None, contextual: bool = True) -> None:
        self.contextual = contextual
        self.messages: list[dict[str, str]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def ask(self, text: str, client: ChatClient) -> str:
        user_message = {"role": "user", "content": text}
        if self.contextual:
            request_messages = [*self.messages, user_message]
        else:
            system_messages = [message for message in self.messages if message["role"] == "system"]
            request_messages = [*system_messages, user_message]

        reply = client.complete(request_messages)
        if self.contextual:
            self.messages.extend((user_message, {"role": "assistant", "content": reply}))
        return reply

    def clear(self) -> None:
        self.messages = [message for message in self.messages if message["role"] == "system"]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.messages, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(
            isinstance(item, dict)
            and item.get("role") in {"system", "user", "assistant"}
            and isinstance(item.get("content"), str)
            for item in data
        ):
            raise ChatError(f"History file is not a valid conversation: {path}")
        self.messages = data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hold a contextual conversation with an LLM.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("EMULLM_BASE_URL", DEFAULT_BASE_URL),
        help=f"OpenAI-compatible API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EMULLM_MODEL", DEFAULT_MODEL),
        help=f"model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key; defaults to OPENAI_API_KEY (not needed by emullm)",
    )
    parser.add_argument("--system", help="system prompt to start a new conversation with")
    parser.add_argument("--history-file", type=Path, help="load and save conversation history as JSON")
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="send no prior turns; exit after an optional command-line prompt",
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="request timeout in seconds")
    parser.add_argument("prompt", nargs="*", help="optional first message")
    return parser


def _print_help() -> None:
    print("Commands: /help, /history, /clear, /quit")


def _show_history(conversation: Conversation) -> None:
    if not conversation.messages:
        print("(conversation is empty)")
        return
    for message in conversation.messages:
        print(f"{message['role']}> {message['content']}")


def _send(text: str, conversation: Conversation, client: ChatClient, history_file: Path | None) -> None:
    try:
        reply = conversation.ask(text, client)
    except ChatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return
    print(f"assistant> {reply}")
    if history_file:
        conversation.save(history_file)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.one_shot and args.history_file:
        print("error: --one-shot cannot be combined with --history-file", file=sys.stderr)
        return 2

    conversation = Conversation(args.system, contextual=not args.one_shot)
    if args.history_file:
        try:
            conversation.load(args.history_file)
        except (ChatError, OSError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    client = ChatClient(args.base_url, args.model, args.api_key, args.timeout)
    print(f"Connected to {client.url} using model {client.model}")
    _print_help()

    if args.prompt:
        _send(" ".join(args.prompt), conversation, client, args.history_file)
        if args.one_shot:
            return 0

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not text:
            continue
        if text in {"/quit", "/exit"}:
            return 0
        if text == "/help":
            _print_help()
            continue
        if text == "/history":
            _show_history(conversation)
            continue
        if text == "/clear":
            conversation.clear()
            if args.history_file:
                conversation.save(args.history_file)
            print("Conversation cleared.")
            continue
        _send(text, conversation, client, args.history_file)


if __name__ == "__main__":
    raise SystemExit(main())
